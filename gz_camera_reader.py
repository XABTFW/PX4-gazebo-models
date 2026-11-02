#!/usr/bin/env python3
"""Gazebo camera tracking; optional DYT SITL telemetry and gimbal following."""

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
from gz.msgs10 import image_pb2
from gz.msgs10.stringmsg_pb2 import StringMsg


def publish_sim_range(publisher, observer, measurement=None, target_id=None):
    """Publish the displayed distance, without refreshing its exposure timestamp."""
    if publisher is None:
        return
    record = dict(measurement or {'valid': False, 'source': 'gazebo_ground_truth'})
    record.update(schema_version=1, observer_model=observer, tracking_target_id=target_id)
    publisher.publish(StringMsg(data=json.dumps(record, allow_nan=False)))


def timestamp_ns(message):
    if not message.HasField("header") or not message.header.HasField("stamp"):
        raise ValueError("missing_image_timestamp")
    stamp = message.header.stamp
    if stamp.sec < 0 or not 0 <= stamp.nsec < 1_000_000_000:
        raise ValueError("invalid_image_timestamp")
    return stamp.sec*1_000_000_000+stamp.nsec


def decode_image(message):
    """Respect per-row padding; return a new contiguous OpenCV BGR image."""
    formats = {
        image_pb2.RGB_INT8: (3, cv2.COLOR_RGB2BGR),
        image_pb2.BGR_INT8: (3, None),
        image_pb2.RGBA_INT8: (4, cv2.COLOR_RGBA2BGR),
        image_pb2.BGRA_INT8: (4, cv2.COLOR_BGRA2BGR),
        image_pb2.L_INT8: (1, cv2.COLOR_GRAY2BGR),
    }
    if message.pixel_format_type not in formats:
        raise ValueError("unsupported_pixel_format:"+str(message.pixel_format_type))
    channels, conversion = formats[message.pixel_format_type]
    width, height, step = message.width, message.height, message.step
    if width <= 0 or height <= 0 or step < width*channels:
        raise ValueError("invalid_image_dimensions_or_step")
    if len(message.data) != step*height:
        raise ValueError("image_payload_length_mismatch")
    pixels = np.frombuffer(message.data, dtype=np.uint8).reshape(height, step)
    pixels = np.ascontiguousarray(pixels[:, :width*channels]).reshape(height, width, channels)
    if conversion is not None:
        return cv2.cvtColor(pixels, conversion)
    return pixels.copy()


class ImageStream:
    """Keep only the newest received image; slow consumers do not build a queue."""

    def __init__(self):
        self.lock = threading.Lock()
        self.message = None
        self.received_at = None
        self.advanced_at = None
        self.stamp = None
        self.sequence = 0
        self.epoch = 0
        self.error = None

    def receive(self, message):
        now = time.monotonic()
        try:
            stamp = timestamp_ns(message)
            error = None
        except ValueError as exception:
            stamp, error = None, str(exception)
        with self.lock:
            self.sequence += 1
            self.message, self.received_at, self.error = message, now, error
            if error is None:
                if self.stamp is None or stamp != self.stamp:
                    if self.stamp is not None and stamp < self.stamp:
                        self.epoch += 1
                    self.advanced_at = now
                self.stamp = stamp

    def snapshot(self, max_age):
        with self.lock:
            if self.message is None:
                raise ValueError("waiting_for_image")
            if self.error is not None:
                raise ValueError(self.error)
            age = time.monotonic()-self.received_at
            if age > max_age:
                raise ValueError("stale_image")
            if time.monotonic()-self.advanced_at > max_age:
                raise ValueError("image_timestamp_not_advancing")
            return self.message, self.sequence, self.epoch, age


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def loss_grace_seconds(value):
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 0.2:
        raise argparse.ArgumentTypeError('must be finite and between 0 and 0.2 seconds')
    return result


def gimbal_rate(value):
    result = positive_float(value)
    if result > 2:
        raise argparse.ArgumentTypeError('must be <= 2 rad/s')
    return result


def gimbal_gain(value):
    result = positive_float(value)
    if result > 20:
        raise argparse.ArgumentTypeError('must be <= 20 /s')
    return result


def match_threshold(value):
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return result


def parse_roi(value):
    try:
        x, y, width, height = map(int, value.split(","))
    except ValueError as exception:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from exception
    if x < 0 or y < 0 or width < 4 or height < 4:
        raise argparse.ArgumentTypeError("ROI needs nonnegative x/y and width/height >= 4")
    return x, y, width, height


class TargetTracker:
    """Motion-predicted matching with guarded resolution upgrades on approach.

    Foreground boundaries and texture validation reject uniform sky matches.
    A missed match never becomes a valid target measurement.
    """

    # Bound all scale/rotation searches, including near-field targets filling the image.
    MATCH_MAX_SIDE = 256

    def __init__(self, auto_lock=True, min_area=100, threshold=0.8, lost_timeout=2.0):
        self.auto_lock = auto_lock
        self.min_area = min_area
        self.threshold = threshold
        self.lost_timeout = lost_timeout
        self.next_id = 1
        self.reset()

    def reset(self):
        self.background = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25,
                                                          detectShadows=False)
        self.frames = 0
        self.shape = None
        self.template = None
        self.template_mask = None
        self.match_template = None
        self.border_gray = 0.
        self.scale = 1.0
        self.angle = 0.0
        self.box = None
        self.target_id = None
        self.last_seen = None
        self.velocity = np.zeros(2, dtype=float)
        self.angular_velocity = 0.
        self.pending_recovery = None
        self.confident_frames = 0
        self.template_updates = 0
        self.last_template_update = None
        self.last_upgrade_sample = None
        self.appearance_votes = deque(maxlen=6)
        self.verification_patch = None
        self.flow_gray = None
        self.flow_points = None
        self.flow_stamp = None
        self.flow_ratio = None
        self.flow_evidence = None
        self.match_candidates = 0
        self.update_reason = 'no_lock'
        self.state = "detecting"

    def capture_appearance(self, image, box):
        """Capture real pixels and their surrounding background, not predictions."""
        x, y, width, height = map(int, box)
        if x < 0 or y < 0 or width < 4 or height < 4 or \
                x+width > image.shape[1] or y+height > image.shape[0]:
            raise ValueError("target_roi_out_of_image")
        template = image[y:y+height, x:x+width].copy()
        # Normalized squared difference is undefined for an all-black template.
        if not np.any(template):
            raise ValueError("target_roi_has_no_appearance")
        self.template, self.box = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), (x, y, width, height)
        # Keep real background around a tight ROI: otherwise a smaller template
        # can match the interior of a larger, uniformly coloured aircraft.
        pad = min(max(4, round(max(width, height)*0.12)), 20, x, y,
                  image.shape[1]-x-width, image.shape[0]-y-height)
        if max(width, height) > self.MATCH_MAX_SIDE:
            # Near-field ROIs already include context; extra padding would
            # require an off-screen full template during normal rotation.
            pad = 0
        context = image[y-pad:y+height+pad, x-pad:x+width+pad]
        self.match_template = cv2.cvtColor(context, cv2.COLOR_BGR2GRAY)
        gray = self.match_template
        self.border_gray = float(np.median(np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))))
        # Ignore uniform background when the ROI includes a small aircraft against sky.
        border = np.concatenate((context[0], context[-1], context[:, 0], context[:, -1]))
        background = np.median(border, axis=0)
        foreground = (np.max(np.abs(context.astype(float)-background), axis=2) > 20).astype(np.uint8)*255
        self.template_mask = (cv2.dilate(foreground, np.ones((7, 7), np.uint8))
                              if 0.02 <= np.mean(foreground > 0) <= 0.8 else None)

    def lock_target(self, image, box, stamp_s):
        self.capture_appearance(image, box)
        self.scale = 1.0
        self.angle = 0.0
        self.target_id = self.next_id
        self.next_id += 1
        self.last_seen = stamp_s
        self.velocity[:] = 0
        self.angular_velocity = 0.
        self.pending_recovery = None
        self.confident_frames = 0
        self.template_updates = 0
        self.last_template_update = stamp_s
        self.last_upgrade_sample = stamp_s
        self.appearance_votes.clear()
        self.flow_evidence = None
        self.match_candidates = 0
        self.verification_patch = self.verification_image(image)
        self.seed_flow(image, stamp_s)
        self.update_reason = 'initial_lock'
        self.state = "locked"

    def verification_image(self, image):
        x, y, width, height = self.box
        patch = cv2.cvtColor(image[y:y+height, x:x+width], cv2.COLOR_BGR2GRAY)
        return cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)

    @staticmethod
    def flow_image(image):
        # Bounded independent evidence; no neural model, GPU, or thread pool.
        height, width = image.shape[:2]
        ratio = min(1., 640/max(height, width))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if ratio < 1:
            gray = cv2.resize(gray, (round(width*ratio), round(height*ratio)), interpolation=cv2.INTER_AREA)
        return gray, np.array([gray.shape[1]/width, gray.shape[0]/height])

    def seed_flow(self, image, stamp_s):
        gray, ratio = self.flow_image(image)
        mask = np.zeros_like(gray)
        x, y, width, height = self.box
        left, top = np.floor(np.array([x, y])*ratio).astype(int)
        right, bottom = np.ceil(np.array([x+width, y+height])*ratio).astype(int)
        mask[max(0, top):bottom, max(0, left):right] = 255
        try:
            self.flow_points = cv2.goodFeaturesToTrack(gray, 48, .015, 3, mask=mask, blockSize=3)
        except cv2.error:
            self.flow_points = None  # Template matching remains authoritative.
        self.flow_gray, self.flow_ratio, self.flow_stamp = gray, ratio, stamp_s

    def estimate_flow(self, image, stamp_s):
        """Forward/backward-checked features propose motion, never a valid lock."""
        self.flow_evidence = None
        if (self.flow_points is None or len(self.flow_points) < 4 or self.flow_stamp is None
                or not 0 < stamp_s-self.flow_stamp <= .15):
            return None
        gray, ratio = self.flow_image(image)
        if gray.shape != self.flow_gray.shape:
            return None
        options = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .02))
        try:
            forward, status, error = cv2.calcOpticalFlowPyrLK(self.flow_gray, gray, self.flow_points, None, **options)
            if forward is None or status is None or error is None:
                return None
            backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.flow_gray, forward, None, **options)
        except cv2.error:
            return None
        if backward is None or back_status is None:
            return None
        original, tracked = self.flow_points.reshape(-1, 2), forward.reshape(-1, 2)
        good = (status.ravel() > 0) & (back_status.ravel() > 0) & (error.ravel() < 25)
        good &= np.linalg.norm(backward.reshape(-1, 2)-original, axis=1) <= .8
        good &= np.all(np.isfinite(tracked), axis=1)
        good &= (tracked[:, 0] >= 0) & (tracked[:, 0] < gray.shape[1])
        good &= (tracked[:, 1] >= 0) & (tracked[:, 1] < gray.shape[0])
        if np.count_nonzero(good) < max(4, math.ceil(len(original)*.65)):
            return None
        try:
            transform, inliers = cv2.estimateAffinePartial2D(original[good], tracked[good],
                               method=cv2.RANSAC, ransacReprojThreshold=1.25, maxIters=80, confidence=.95)
        except cv2.error:
            return None
        if transform is None or inliers is None or np.mean(inliers) < .75:
            return None
        scale = float(np.linalg.norm(transform[:, 0]))
        if not np.all(np.isfinite(transform)) or not .75 <= scale <= 1.5:
            return None
        x, y, width, height = self.box
        center = np.array([x+width/2, y+height/2])*self.flow_ratio
        moved = (transform[:, :2]@center+transform[:, 2])/ratio
        angle = math.degrees(math.atan2(-transform[1, 0], transform[0, 0]))
        if abs(angle) > 25:
            return None
        self.flow_evidence = dict(center=moved, scale=scale, angle=angle,
                                  points=int(np.count_nonzero(good)))
        return self.flow_evidence

    def upgrade_appearance(self, image, stamp_s, score, displacement, was_tracking):
        # Real-flight scores often hover around .82-.86. Requiring THREE .88
        # frames starved adaptation entirely. Use independently checked flow or
        # adjacent real-image consistency, with a bounded 4-of-6 vote window.
        if self.last_upgrade_sample is not None and stamp_s <= self.last_upgrade_sample:
            self.update_reason = 'repeated_or_reversed_exposure_not_learned'
            return
        gap = stamp_s-self.last_upgrade_sample if self.last_upgrade_sample is not None else float('inf')
        self.last_upgrade_sample = stamp_s
        x, y, width, height = self.box
        confident = (was_tracking and score >= max(.88, self.threshold+.06)
                     and displacement <= max(12, min(width, height)*.4))
        self.confident_frames = (self.confident_frames+1 if gap <= .15 else 1) if confident else 0
        patch = self.verification_image(image)
        adjacent = (self.verification_patch is not None
                    and float(cv2.matchTemplate(patch, self.verification_patch, cv2.TM_SQDIFF_NORMED)[0, 0]) <= .12)
        if adjacent:
            agreement = float(cv2.matchTemplate(self.edges(patch), self.edges(self.verification_patch),
                                               cv2.TM_CCORR_NORMED)[0, 0])
            adjacent = math.isfinite(agreement) and agreement >= .65
        continuous = (was_tracking and score >= max(.815, self.threshold+.015)
                      and displacement <= max(12, min(width, height)*.4)
                      and (self.flow_evidence is not None or adjacent))
        self.appearance_votes.append((stamp_s, bool(continuous)))
        votes = sum(ok and 0 <= stamp_s-stamp <= .4 for stamp, ok in self.appearance_votes)
        supported = continuous and votes >= 4
        resolution_growth = self.scale >= 1.6
        changed_view = (abs(self.angle) >= 15 or score < .9) and stamp_s-self.last_template_update >= .6
        self.update_reason = 'insufficient_independent_confirmation'
        if not (self.confident_frames >= 3 or supported):
            return
        if not (resolution_growth or changed_view):
            self.update_reason = 'appearance_still_stable'
            return
        if min(width, height) < 12:
            self.update_reason = 'awaiting_useful_resolution'
            return
        if stamp_s-self.last_template_update < .25:
            self.update_reason = 'update_cooldown'
            return
        if x <= 2 or y <= 2 or x+width >= image.shape[1]-2 or y+height >= image.shape[0]-2:
            self.update_reason = 'clipped_target_not_learned'
            return
        patch = cv2.cvtColor(image[y:y+height, x:x+width], cv2.COLOR_BGR2GRAY)
        if np.std(patch) < 8 or np.count_nonzero(cv2.Canny(patch, 30, 90)) < 20:
            self.update_reason = 'insufficient_target_texture'
            return
        self.capture_appearance(image, self.box)
        self.scale, self.angle, self.angular_velocity = 1., 0., 0.
        self.template_updates += 1
        self.last_template_update = stamp_s
        self.confident_frames = 0
        self.appearance_votes.clear()
        self.update_reason = 'updated_from_confirmed_pixels'

    def appearance(self, scale, angle):
        source = self.match_template
        height, width = source.shape
        if abs(angle) < 1e-6:
            size = (max(4, round(width*scale)), max(4, round(height*scale)))
            mask = None if self.template_mask is None else cv2.resize(
                self.template_mask, size, interpolation=cv2.INTER_NEAREST)
            return cv2.resize(source, size, interpolation=(cv2.INTER_AREA if scale < 1.0
                                                                  else cv2.INTER_LINEAR)), mask
        radians = math.radians(angle)
        size = (max(4, round(scale*(width*abs(math.cos(radians))+height*abs(math.sin(radians))))),
                max(4, round(scale*(height*abs(math.cos(radians))+width*abs(math.sin(radians))))))
        transform = cv2.getRotationMatrix2D(((width-1)/2, (height-1)/2), angle, scale)
        transform[:, 2] += [(size[0]-width)/2, (size[1]-height)/2]
        template = cv2.warpAffine(source, transform, size, borderValue=self.border_gray)
        original_mask = self.template_mask if self.template_mask is not None else np.full_like(source, 255)
        mask = cv2.warpAffine(original_mask, transform, size, flags=cv2.INTER_NEAREST)
        return template, mask

    @staticmethod
    def edges(gray):
        # One-pixel tolerance to blur/antialiasing, still penalising extra or
        # missing aircraft boundaries. Operates only on bounded search images.
        return cv2.dilate(cv2.Canny(gray, 30, 90), np.ones((3, 3), np.uint8)).astype(np.float32)/255

    def match(self, image, stamp_s):
        self.match_candidates = 0
        x, y, width, height = self.box
        elapsed = max(0., stamp_s-self.last_seen)
        center = np.array([x+width/2, y+height/2])
        predicted = center+self.velocity*min(elapsed, .2)
        flow = self.estimate_flow(image, stamp_s)
        if flow is not None:
            predicted = flow['center']
        # Search both last observed and predicted locations. Enlarge quickly
        # on a miss, but keep compute bounded by MATCH_MAX_SIDE below.
        base = max(32, min(80, max(width, height)*.4))
        margin = base+min(160, elapsed*240)
        left = max(0, math.floor(min(center[0], predicted[0])-width/2-margin))
        top = max(0, math.floor(min(center[1], predicted[1])-height/2-margin))
        right = min(image.shape[1], math.ceil(max(center[0], predicted[0])+width/2+margin))
        bottom = min(image.shape[0], math.ceil(max(center[1], predicted[1])+height/2+margin))
        search = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        # Very large targets need fewer pixels for identity/outline checks;
        # keep rotation fallback fast enough for the 30 Hz camera.
        # Near-field outlines do not need a 256px search pyramid. Preserve
        # full resolution for tiny distant targets, where every pixel matters.
        max_side = 192 if max(width, height) >= 96 else self.MATCH_MAX_SIDE
        match_scale = min(1.0, max_side/max(search.shape))
        if match_scale < 1.0:
            search = cv2.resize(search, None, fx=match_scale, fy=match_scale, interpolation=cv2.INTER_AREA)
        ratio_x, ratio_y = search.shape[1]/(right-left), search.shape[0]/(bottom-top)
        if int(np.max(search))-int(np.min(search)) < 6:
            # Such a range cannot pass the existing >=3 standard-deviation
            # texture gate. Avoid expensive rotation retries on empty sky.
            return None, predicted
        pixels = search.astype(np.float32)
        squared_pixels = pixels*pixels
        search_edges = self.edges(search)
        # Rate-bounded scale proposals prevent exponential frame-by-frame
        # collapse. True scale changes remain allowed across successive images.
        step = min(.08, max(.025, elapsed*1.5))
        scale_center = self.scale*(flow['scale'] if flow is not None else 1.)
        scales = tuple(scale_center*math.exp(v*step) for v in (0, -1, 1, -.5, .5))
        # Predicted angular motion already accounts for the main rotation.
        # Retain the +/-8 degree recovery range; omit redundant +/-1 searches.
        rotations = (0, -2, 2, -4, 4, -8, 8)
        predicted_angle = self.angle+float(np.clip(self.angular_velocity*min(elapsed, .1), -8, 8))
        if flow is not None:
            predicted_angle = self.angle+flow['angle']
        best = None
        variance_cache = {}
        for rotation in rotations:
            if rotation and best is not None and best['quality'] >= max(.86, self.threshold+.04):
                break
            if (rotation and flow is not None and best is not None
                    and best['quality'] >= self.threshold+.02
                    and np.linalg.norm(best['center']-predicted) <= max(4, min(width, height)*.1)
                    and abs(math.log(best['scale']/scale_center)) <= step+1e-9):
                # Two independent current-image cues agree: no need to scan
                # the other six angle groups just to gain cosmetic confidence.
                break
            for scale in scales:
                # Do not turn a normal-size target into an unresolvable 4px
                # speck. Existing tiny ROIs keep their original minimum size.
                minimum = min(6, min(self.template.shape))
                if min(self.template.shape)*scale < minimum:
                    continue
                angle = predicted_angle+rotation
                radians = math.radians(angle)
                original_h, original_w = self.template.shape
                bbox_w = round(scale*(original_w*abs(math.cos(radians))+original_h*abs(math.sin(radians))))
                bbox_h = round(scale*(original_h*abs(math.cos(radians))+original_w*abs(math.sin(radians))))
                if bbox_w > image.shape[1] or bbox_h > image.shape[0]:
                    continue
                template, mask = self.appearance(scale*match_scale, angle)
                th, tw = template.shape
                if tw > search.shape[1] or th > search.shape[0]:
                    continue
                matches = cv2.matchTemplate(search, template, cv2.TM_SQDIFF_NORMED, mask=mask)
                self.match_candidates += 1
                matches[~np.isfinite(matches)] = 1.0
                if (tw, th) not in variance_cache:
                    mean = cv2.boxFilter(pixels, -1, (tw, th), anchor=(0, 0))
                    mean_square = cv2.boxFilter(squared_pixels, -1, (tw, th), anchor=(0, 0))
                    variance_cache[tw, th] = mean_square-mean*mean
                variance = variance_cache[tw, th][:matches.shape[0], :matches.shape[1]]
                matches[variance < max(3., float(np.std(template))*.25)**2] = 1.0
                outline = self.edges(template)
                if np.count_nonzero(outline) >= 12:
                    agreement = cv2.matchTemplate(search_edges, outline, cv2.TM_CCORR_NORMED)
                    agreement[~np.isfinite(agreement)] = 0
                    # Raw pixel similarity alone cannot validate a boundary
                    # mismatch. The public threshold is not lowered.
                    quality = (1-np.minimum(matches, 1))*(.65+.35*np.clip(agreement, 0, 1))
                    quality[agreement < .45] = 0
                else:
                    quality = 1-np.minimum(matches, 1)
                # Soft continuity priors: colour/shape evidence must improve
                # enough before scale or rotation is allowed to drift.
                penalty = .15*abs(math.log(scale/scale_center))+.0008*abs(rotation)
                candidate_x = left+(np.arange(quality.shape[1])+tw/2)/ratio_x
                candidate_y = top+(np.arange(quality.shape[0])+th/2)/ratio_y
                distance = np.hypot(candidate_x[None, :]-predicted[0], candidate_y[:, None]-predicted[1])
                position_penalty = .015*np.minimum(distance/max(24, width, height), 3)
                rank = (quality-penalty-position_penalty).astype(np.float32)
                _, value, _, location = cv2.minMaxLoc(rank)
                if best is None or value > best['rank']:
                    ix, iy = location
                    refined = [float(ix), float(iy)]
                    for axis in (0, 1):
                        if 0 < ix < rank.shape[1]-1 and 0 < iy < rank.shape[0]-1:
                            before = float(rank[iy-(axis == 1), ix-(axis == 0)])
                            after = float(rank[iy+(axis == 1), ix+(axis == 0)])
                            curvature = before+after-2*value
                            if before > 0 and after > 0 and curvature < -1e-6:
                                refined[axis] += float(np.clip((before-after)/(2*curvature), -.5, .5))
                    best = {'rank': value, 'quality': float(quality[iy, ix]), 'scale': scale, 'angle': angle,
                            'center': np.array([left+(refined[0]+tw/2)/ratio_x,
                                                top+(refined[1]+th/2)/ratio_y])}
        return best, predicted

    def update(self, image, stamp_s, roi=None):
        if self.shape is not None and self.shape != image.shape:
            self.reset()
        self.shape = image.shape
        if roi is not None:
            self.lock_target(image, roi, stamp_s)
        self.frames += 1
        candidates = []
        mask = self.background.apply(image) if self.template is None else None
        # MOG2 marks the entire initial frame as foreground. Warm up first.
        if self.frames > 10 and mask is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                if self.min_area <= cv2.contourArea(contour) <= image.shape[0]*image.shape[1]*0.5 \
                        and width >= 4 and height >= 4 and np.any(image[y:y+height, x:x+width]):
                    candidates.append((x, y, width, height))
            candidates.sort(key=lambda box: box[2]*box[3], reverse=True)

        score = None
        valid = False
        if roi is not None:
            score, valid = 1.0, True
        elif self.template is None and self.state == "detecting" and self.auto_lock and candidates:
            self.lock_target(image, candidates[0], stamp_s)
            score, valid = 1.0, True
        elif self.template is not None and self.state != "lost":
            elapsed = stamp_s-self.last_seen
            if elapsed > self.lost_timeout:
                self.state = "lost"
                self.pending_recovery = None
            else:
                best, predicted = self.match(image, stamp_s)
                score = 0.0 if best is None else best['quality']
                accepted = score >= self.threshold
                if accepted and self.state == 'searching':
                    # A distant reacquisition needs two genuinely matching
                    # frames. Never publish a predicted position as a target.
                    width, height = self.box[2:]
                    distant = np.linalg.norm(best['center']-predicted) > max(32, max(width, height)*.75)
                    previous = self.pending_recovery
                    confirmed = (previous is not None and 0 < stamp_s-previous[0] <= .15
                                 and np.linalg.norm(best['center']-previous[1]) <= max(24, max(width, height)*.5)
                                 and abs(math.log(best['scale']/previous[2])) <= .12)
                    if distant and not confirmed:
                        self.pending_recovery = (stamp_s, best['center'], best['scale'])
                        accepted = False
                if accepted:
                    was_tracking = self.state in ('tracking', 'locked')
                    old_center = np.array([self.box[0]+self.box[2]/2, self.box[1]+self.box[3]/2])
                    cx, cy = best['center']
                    if 0 < elapsed <= .2:
                        measured_velocity = np.clip((best['center']-old_center)/elapsed, -2500, 2500)
                        self.velocity = .5*self.velocity+.5*measured_velocity
                        angle_rate = np.clip((best['angle']-self.angle)/elapsed, -240, 240)
                        self.angular_velocity = .3*self.angular_velocity+.7*angle_rate
                    else:
                        self.velocity[:] = 0
                        self.angular_velocity = 0.
                    self.pending_recovery = None
                    self.scale, self.angle = best['scale'], best['angle']
                    radians = math.radians(self.angle)
                    original_h, original_w = self.template.shape
                    width = max(4, round(self.scale*(original_w*abs(math.cos(radians))+
                                                   original_h*abs(math.sin(radians)))))
                    height = max(4, round(self.scale*(original_h*abs(math.cos(radians))+
                                                    original_w*abs(math.sin(radians)))))
                    self.box = (max(0, min(image.shape[1]-width, round(cx-width/2))),
                                max(0, min(image.shape[0]-height, round(cy-height/2))), width, height)
                    self.last_seen = stamp_s
                    self.state, valid = "tracking", True
                    self.upgrade_appearance(image, stamp_s, score,
                                            float(np.linalg.norm(best['center']-predicted)), was_tracking)
                    self.verification_patch = self.verification_image(image)
                    self.seed_flow(image, stamp_s)
                else:
                    self.confident_frames = 0
                    self.appearance_votes.append((stamp_s, False))
                    self.update_reason = 'no_valid_visual_match'
                    if score < self.threshold:
                        self.pending_recovery = None
                    self.state = "searching"

        result = {"state": self.state, "valid": valid, "target_id": self.target_id,
                  "bbox_xywh": list(self.box) if valid else None,
                  "match_score": score, "candidates_xywh": [list(box) for box in candidates]}
        result['scale'] = self.scale
        result['template_updates'] = self.template_updates
        result['template_update_reason'] = self.update_reason
        result['matching_candidates'] = self.match_candidates
        result['flow_support_points'] = self.flow_evidence['points'] if self.flow_evidence else 0
        result['rotation_deg'] = self.angle
        result['recovery_pending'] = self.pending_recovery is not None and not valid
        if valid:
            x, y, width, height = self.box
            cx, cy = x+width/2, y+height/2
            result["center_px"] = [cx, cy]
            result["center_error_normalized"] = [2*cx/image.shape[1]-1, 2*cy/image.shape[0]-1]
        return result


def draw_tracking(preview, target, sim_range=None):
    for x, y, width, height in target["candidates_xywh"]:
        cv2.rectangle(preview, (x, y), (x+width, y+height), (0, 200, 255), 1)
    if target["valid"]:
        x, y, width, height = target["bbox_xywh"]
        cv2.rectangle(preview, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cx, cy = map(int, target["center_px"])
        cv2.drawMarker(preview, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        if sim_range is not None:
            measured = sim_range.get('valid') and sim_range.get('distance_m') is not None
            label = f"SIM distance: {sim_range['distance_m']:.2f} m" if measured else 'SIM distance: --'
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, .55, 2
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            tx = max(0, min(x, preview.shape[1]-tw-4))
            ty = max(th+baseline+4, min(y-8, preview.shape[0]-baseline-3))
            cv2.rectangle(preview, (tx, ty-th-3), (min(preview.shape[1]-1, tx+tw+3), ty+baseline), (0, 0, 0), -1)
            cv2.putText(preview, label, (tx+1, ty), font, scale,
                        (0, 255, 0) if measured else (0, 200, 255), thickness)
    label = f"target={target['target_id']} {target['state']}"
    cv2.putText(preview, label, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0) if target["valid"] else (0, 200, 255), 2)


def window_closed(window):
    try:
        visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
        if visible >= 0:
            return visible == 0
        # GTK3 returns -1 for unsupported VISIBLE even while the window is open.
        return cv2.getWindowProperty(window, cv2.WND_PROP_ASPECT_RATIO) < 0
    except cv2.error:
        # Some backends raise when querying a window destroyed by its close button.
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-topics", action="store_true", help="list actual Gazebo topics, then exit")
    parser.add_argument("--topic", help="exact image topic from gz topic -l; type must be gz.msgs.Image")
    parser.add_argument("--headless", action="store_true", help="log/save without opening a window")
    parser.add_argument("--record-dir", type=Path, help="create a NEW directory for PNGs and frames.jsonl")
    parser.add_argument("--rate", type=positive_float, default=60.0, help="maximum processing/saving wall Hz (default 60; camera is 30 Hz)")
    parser.add_argument("--max-age", type=positive_float, default=0.5, help="receive freshness limit, wall seconds")
    parser.add_argument("--duration", type=positive_float, help="stop after this many wall seconds")
    parser.add_argument("--manual-lock", action="store_true", help="detect but only lock a mouse/CLI ROI")
    parser.add_argument("--roi", type=parse_roi, help="lock first frame ROI: x,y,width,height (also headless)")
    parser.add_argument("--min-target-area", type=positive_int, default=100, help="minimum motion contour pixels")
    parser.add_argument("--match-threshold", type=match_threshold, default=0.8, help="minimum appearance match score")
    parser.add_argument("--lost-timeout", type=positive_float, default=2.0, help="abandon missing target after simulation seconds")
    parser.add_argument("--dyt-model", help="publish tracking to this fpv_quadtailsitter_gimbal SITL instance")
    parser.add_argument('--range-target', default='x500_2',
                        help='Gazebo model associated with the locked ROI for ground-truth distance overlay (default x500_2)')
    parser.add_argument("--world", default="default", help="Gazebo world for DYT pose feedback")
    parser.add_argument("--follow-gimbal", action="store_true", help="with --dyt-model, center target using simulated gimbal joints")
    parser.add_argument('--sim-loss-grace', type=loss_grace_seconds, default=0.2,
                        help='delay DYT loss notifications for brief misses; no valid frames are replayed (0 disables)')
    parser.add_argument('--sim-native-frame', action='store_true',
                        help='disable SITL virtual-frame reprojection (native pitch >90 deg is rejected by PX4)')
    parser.add_argument('--sim-gimbal-rate', type=gimbal_rate, default=1.0,
                        help='simulated visual-follow joint speed limit, rad/s (default 1, maximum 2)')
    parser.add_argument('--sim-gimbal-gain', type=gimbal_gain, default=10.0,
                        help='simulated image centering response, /s (default 10, maximum 20; old response 5)')
    args = parser.parse_args()
    if not args.list_topics and not args.topic:
        parser.error("--topic is required unless --list-topics is used")
    if args.dyt_model and args.rate < 30:
        parser.error("DYT input requires --rate >= 30 to consume the 30 Hz camera")
    if args.follow_gimbal and not args.dyt_model:
        parser.error("--follow-gimbal requires --dyt-model")
    if not args.list_topics and not args.headless and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("no graphical session; use --headless")
    # Avoid large OpenCV thread pools causing latency spikes alongside Gazebo rendering.
    cv2.setNumThreads(1)
    from gz.transport13 import Node
    node = Node()
    manifest = None
    dyt_bridge = None
    range_publisher = None
    window = "Gazebo camera - drag to lock, r to reset, q/Esc to exit"
    window_open = False
    try:
        if args.list_topics:
            # Allow discovery briefly; a topic name alone does not prove its type.
            time.sleep(0.5)
            print("\n".join(sorted(node.topic_list())))
            return 0
        if args.dyt_model:
            from gz_dyt_bridge import DytBridge, check_target_texture, simulation_distance
            dyt_bridge = DytBridge(node, args.dyt_model, args.world, args.topic, args.follow_gimbal,
                                   loss_grace=args.sim_loss_grace, virtual_frame=not args.sim_native_frame,
                                   gimbal_rate=args.sim_gimbal_rate, gimbal_gain=args.sim_gimbal_gain)
            range_publisher = node.advertise(f'/model/{args.dyt_model}/dyt/sim_range', StringMsg)
            if not range_publisher:
                raise ValueError('simulation_range_advertise_failed')
        if args.record_dir:
            args.record_dir.mkdir(parents=True, exist_ok=False)
            manifest = (args.record_dir/"frames.jsonl").open("x", encoding="utf-8")

        def emit(record):
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n"
            sys.stdout.write(line)
            sys.stdout.flush()
            if manifest:
                manifest.write(line)
                manifest.flush()

        stream = ImageStream()
        if not node.subscribe(image_pb2.Image, args.topic, stream.receive):
            raise ValueError("image_subscription_failed")
        emit({"type": "metadata", "schema_version": 2, "image_topic": args.topic,
              "read_only": dyt_bridge is None, "dyt_model": args.dyt_model,
              "follow_gimbal": args.follow_gimbal, "saved_pixel_format": "BGR_INT8_as_PNG",
              "sim_loss_grace_s": args.sim_loss_grace,
              "sim_virtual_frame": not args.sim_native_frame,
              "range_source": 'gazebo_ground_truth' if dyt_bridge else None,
              "range_target_model": args.range_target if dyt_bridge else None,
              "range_topic": f'/model/{args.dyt_model}/dyt/sim_range' if dyt_bridge else None,
              "sim_gimbal_rate_rad_s": args.sim_gimbal_rate,
              "sim_gimbal_gain_per_s": args.sim_gimbal_gain,
              "alpha_handling": "discarded_for_RGBA_BGRA", "geometry_pose_input": dyt_bridge is not None,
              "max_processing_rate_wall_hz": args.rate, "queue_policy": "latest_received_only",
              "timestamp_source": "gz.msgs.Image.header.stamp", "freshness": "local_receive_wall_time",
              "detector": "MOG2_motion", "tracker": "context_edge_motion_scale_rotation_template",
              "template_resolution_upgrade": "four_of_six_visual_matches_with_flow_or_pixel_consistency",
              "motion_evidence": "bounded_forward_backward_LK_RANSAC_not_valid_by_itself",
              "matching_max_side_px": TargetTracker.MATCH_MAX_SIDE,
              "near_matching_max_side_px": 192,
              "auto_lock": not args.manual_lock, "initial_roi": args.roi,
              "min_target_area": args.min_target_area, "match_threshold": args.match_threshold,
              "lost_timeout_sim_s": args.lost_timeout, "motion_warmup_frames": 10,
              "dyt_max_frame_gap_sim_s": 0.12, "dyt_max_image_age_sim_s": 0.25})
        tracker = TargetTracker(not args.manual_lock, args.min_target_area,
                                args.match_threshold, args.lost_timeout)
        pending_roi = args.roi
        drag_start = None

        def select_target(event, x, y, flags, userdata):
            nonlocal pending_roi, drag_start
            if event == cv2.EVENT_LBUTTONDOWN:
                drag_start = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
                sx, sy = drag_start
                drag_start = None
                if abs(x-sx) >= 4 and abs(y-sy) >= 4:
                    pending_roi = (min(sx, x), min(sy, y), abs(x-sx), abs(y-sy))

        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window, select_target)
            window_open = True
        start = time.monotonic()
        last_sequence = 0
        last_state = None
        last_bridge_error = None
        dyt_timing = None
        sim_range = None
        processed = 0
        last_shape = (360, 640, 3)
        last_epoch = None
        last_stamp = None
        tracking_elapsed = 0.0
        while args.duration is None or time.monotonic()-start < args.duration:
            tick = time.monotonic()
            preview = None
            try:
                message, sequence, epoch, age = stream.snapshot(args.max_age)
                if sequence != last_sequence:
                    tracking_start = time.monotonic()
                    image = decode_image(message)
                    stamp = timestamp_ns(message)
                    if last_epoch is not None and epoch != last_epoch:
                        tracker.reset()
                        last_stamp = None
                        if dyt_bridge:
                            dyt_bridge.invalidate(stamp_ns=stamp, force=True)
                    if tracker.shape is not None and tracker.shape != image.shape:
                        tracker.reset()
                        last_stamp = None
                        if dyt_bridge:
                            dyt_bridge.invalidate(stamp_ns=stamp, force=True)
                    # Repeated transport messages with a frozen stamp are not new tracker frames.
                    if stamp != last_stamp:
                        if pending_roi is not None and dyt_bridge:
                            # Selecting a different target is intentional, not
                            # a brief tracking miss of the previous target.
                            dyt_bridge.invalidate(force=True)
                            publish_sim_range(range_publisher, args.dyt_model)
                        try:
                            target = tracker.update(image, stamp/1e9, pending_roi)
                        except ValueError as exception:
                            if pending_roi is None:
                                raise
                            emit({"type": "selection_error", "error": str(exception)})
                            target = tracker.update(image, stamp/1e9)
                        tracking_elapsed = time.monotonic()-tracking_start
                        pending_roi = None
                        last_stamp = stamp
                        if dyt_bridge:
                            sim_range = simulation_distance(dyt_bridge, message, target, args.range_target)
                            # Telemetry/pose outages must not erase the selected image identity.
                            try:
                                check_target_texture(image, target)
                                dyt_timing = dyt_bridge.update(message, target)
                                bridge_error = None
                            except ValueError as exception:
                                bridge_error = str(exception)
                                dyt_timing = dyt_bridge.invalidate(stamp_ns=stamp)
                            publish_sim_range(range_publisher, args.dyt_model,
                                              sim_range if bridge_error is None else None, target.get('target_id'))
                            if bridge_error != last_bridge_error:
                                emit({"type": "dyt_state", "valid": bridge_error is None,
                                      "error": bridge_error})
                                last_bridge_error = bridge_error
                    last_epoch = epoch
                    last_shape = image.shape
                    processed += 1
                    record = {"type": "frame", "valid": True, "sequence": sequence,
                              "processed_frames": processed, "sim_epoch": epoch,
                              "stamp_ns": stamp, "width": message.width,
                              "height": message.height, "source_step_bytes": message.step,
                              "source_pixel_format": image_pb2.PixelFormatType.Name(message.pixel_format_type),
                              "receive_age_wall_s": age, "wall_elapsed_s": tick-start,
                              "skipped_received_frames": sequence-last_sequence-1, "target": target,
                              "tracking_time_wall_s": tracking_elapsed,
                              "processing_time_wall_s": time.monotonic()-tick,
                              "dyt_timing": dyt_timing,
                              "sim_range": sim_range,
                              "dyt_valid": dyt_bridge is not None and target["valid"] and last_bridge_error is None}
                    if args.record_dir:
                        filename = f"frame_{sequence:09d}.png"
                        if not cv2.imwrite(str(args.record_dir/filename), image):
                            raise OSError("PNG write failed")
                        record["png"] = filename
                    emit(record)
                    last_sequence = sequence
                    # Saved PNGs contain pixels only. Overlay a separate display copy.
                    preview = image.copy()
                    cv2.putText(preview, f"seq={sequence} sim={record['stamp_ns']/1e9:.3f}s", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    draw_tracking(preview, target, sim_range)
                state = "receiving_image"
            except ValueError as exception:
                state = str(exception)
                # Keep the template for recovery; publish no valid measurement while stale.
                if dyt_bridge:
                    dyt_bridge.invalidate()
                    publish_sim_range(range_publisher, args.dyt_model)
                # Clear a stale display instead of presenting the last image as live.
                preview = np.zeros(last_shape, dtype=np.uint8)
                cv2.putText(preview, state, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if state != last_state:
                emit({"type": "stream_state", "valid": state == "receiving_image",
                      "state": state, "wall_elapsed_s": tick-start})
                print(state, file=sys.stderr)
                last_state = state
            if window_open:
                if preview is not None:
                    cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    tracker.reset()
                    pending_roi = None
                    last_stamp = None
                    if dyt_bridge:
                        dyt_bridge.invalidate(force=True)
                        publish_sim_range(range_publisher, args.dyt_model)
                if window_closed(window):
                    break
            time.sleep(max(0.0, 1/args.rate-(time.monotonic()-tick)))
        return 0
    except (ValueError, OSError, cv2.error) as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        if args.topic:
            node.unsubscribe(args.topic)
        if manifest:
            manifest.close()
        if dyt_bridge:
            publish_sim_range(range_publisher, args.dyt_model)
            dyt_bridge.close()
        if window_open:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
