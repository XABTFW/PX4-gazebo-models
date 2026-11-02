#!/usr/bin/env python3
"""Image decoding and headless CLI checks; no live Gazebo or GUI required."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
from gz.msgs10 import image_pb2

import gz_camera_reader as reader
from gz_net_physics import CameraRangeGate


class RangeConnectionTest(unittest.TestCase):
    def test_display_measurement_reaches_gate_with_original_timestamp(self):
        publisher = MagicMock()
        measurement = dict(valid=True, source='gazebo_ground_truth', target_model='target',
                           distance_m=2.1, stamp_ns=10_000_000_000)
        reader.publish_sim_range(publisher, 'observer', measurement, 3)
        record = json.loads(publisher.publish.call_args.args[0].data)
        self.assertNotIn('observer_model', measurement)
        gate = CameraRangeGate('observer', 'target')
        with patch('gz_net_physics.time.monotonic', return_value=100):
            gate.observe(record)
            self.assertEqual(gate.measurement(10.1)['distance_m'], 2.1)
            self.assertFalse(gate.measurement(10.3)['valid'])
            self.assertFalse(gate.measurement(9.9)['valid'])
        with patch('gz_net_physics.time.monotonic', return_value=100.3):
            self.assertFalse(gate.measurement(10.1)['valid'])
        reader.publish_sim_range(publisher, 'observer')
        gate.observe(json.loads(publisher.publish.call_args.args[0].data))
        self.assertFalse(gate.measurement(10.1)['valid'])

    def test_mismatched_or_invalid_measurements_clear_previous_range(self):
        gate = CameraRangeGate('observer', 'target')
        good = dict(schema_version=1, valid=True, source='gazebo_ground_truth',
                    observer_model='observer', target_model='target', tracking_target_id=1,
                    distance_m=2., stamp_ns=10_000_000_000)
        for field, value in [('valid', False), ('source', 'laser'), ('observer_model', 'other'),
                             ('target_model', 'other'), ('tracking_target_id', None),
                             ('distance_m', float('nan')), ('distance_m', True),
                             ('distance_m', -1), ('stamp_ns', True), ('schema_version', 2)]:
            with self.subTest(field=field, value=value):
                gate.observe(good)
                self.assertTrue(gate.measurement(10.1)['valid'])
                gate.observe(dict(good, **{field: value}))
                self.assertFalse(gate.measurement(10.1)['valid'])


def image_message():
    image = image_pb2.Image(width=2, height=2, step=8, pixel_format_type=image_pb2.RGB_INT8)
    # Red/blue then green/yellow; two padding bytes per row are not pixels.
    image.data = bytes([255, 0, 0, 0, 0, 255, 100, 100, 0, 255, 0, 255, 255, 0, 100, 100])
    image.header.stamp.sec = 12
    image.header.stamp.nsec = 345
    return image


class WindowTest(unittest.TestCase):
    def test_range_label_is_on_display_copy_and_clamped_inside_image(self):
        image = np.zeros((120, 320, 3), dtype=np.uint8)
        target = {'valid': True, 'target_id': 1, 'state': 'tracking', 'bbox_xywh': [300, 0, 20, 20],
                  'center_px': [310, 10], 'candidates_xywh': []}
        with patch.object(reader.cv2, 'putText', wraps=reader.cv2.putText) as draw:
            reader.draw_tracking(image, target, {'valid': True, 'distance_m': 12.345})
        label = next(call for call in draw.call_args_list if call.args[1].startswith('SIM distance:'))
        self.assertEqual(label.args[1], 'SIM distance: 12.35 m')
        self.assertGreaterEqual(label.args[2][0], 0)
        self.assertGreater(label.args[2][1], 0)
        self.assertLess(label.args[2][0], 320)

    def test_invalid_range_does_not_show_an_old_number(self):
        image = np.zeros((120, 320, 3), dtype=np.uint8)
        target = {'valid': True, 'target_id': 1, 'state': 'tracking', 'bbox_xywh': [100, 50, 20, 20],
                  'center_px': [110, 60], 'candidates_xywh': []}
        with patch.object(reader.cv2, 'putText') as draw:
            reader.draw_tracking(image, target, {'valid': False, 'distance_m': None})
        self.assertIn('SIM distance: --', [call.args[1] for call in draw.call_args_list])

    def test_gtk_unsupported_visibility_keeps_preview_open(self):
        with patch.object(reader.cv2, "getWindowProperty", side_effect=[-1, 16/9]):
            self.assertFalse(reader.window_closed("preview"))

    def test_closed_window_for_supported_and_gtk_backends(self):
        for properties in ([0], [-1, -1]):
            with self.subTest(properties=properties), \
                    patch.object(reader.cv2, "getWindowProperty", side_effect=properties):
                self.assertTrue(reader.window_closed("preview"))
        with patch.object(reader.cv2, "getWindowProperty", side_effect=cv2.error("closed window")):
            self.assertTrue(reader.window_closed("preview"))


class DecodeTest(unittest.TestCase):
    def test_rgb_padding_and_contiguity(self):
        decoded = reader.decode_image(image_message())
        self.assertEqual(decoded.tolist(), [[[0, 0, 255], [255, 0, 0]], [[0, 255, 0], [0, 255, 255]]])
        self.assertTrue(decoded.flags.c_contiguous)

    def test_other_supported_formats(self):
        cases = [(image_pb2.BGR_INT8, [1, 2, 3], [1, 2, 3]),
                 (image_pb2.RGBA_INT8, [1, 2, 3, 4], [3, 2, 1]),
                 (image_pb2.BGRA_INT8, [1, 2, 3, 4], [1, 2, 3]),
                 (image_pb2.L_INT8, [7], [7, 7, 7])]
        for fmt, data, expected in cases:
            with self.subTest(format=fmt):
                image = image_pb2.Image(width=1, height=1, step=len(data), pixel_format_type=fmt, data=bytes(data))
                self.assertEqual(reader.decode_image(image)[0, 0].tolist(), expected)

    def test_bad_dimensions_step_payload_and_format(self):
        for field, value in (("width", 0), ("height", 0), ("step", 1), ("data", b"short"),
                             ("pixel_format_type", image_pb2.R_FLOAT32)):
            with self.subTest(field=field):
                image = image_message()
                setattr(image, field, value)
                with self.assertRaises(ValueError):
                    reader.decode_image(image)

    def test_timestamp_is_preserved_and_missing_rejected(self):
        self.assertEqual(reader.timestamp_ns(image_message()), 12_000_000_345)
        with self.assertRaisesRegex(ValueError, "missing_image_timestamp"):
            reader.timestamp_ns(image_pb2.Image())


class StreamTest(unittest.TestCase):
    def test_wait_stale_frozen_reset_and_invalid_latest(self):
        stream = reader.ImageStream()
        with self.assertRaisesRegex(ValueError, "waiting"):
            stream.snapshot(0.5)
        with patch.object(reader.time, "monotonic", return_value=100):
            stream.receive(image_message())
            self.assertEqual(stream.snapshot(0.5)[2], 0)
        with patch.object(reader.time, "monotonic", return_value=101):
            with self.assertRaisesRegex(ValueError, "stale_image"):
                stream.snapshot(0.5)
            stream.receive(image_message())
            with self.assertRaisesRegex(ValueError, "not_advancing"):
                stream.snapshot(0.5)
            reset = image_message()
            reset.header.stamp.sec = 0
            stream.receive(reset)
            self.assertEqual(stream.snapshot(0.5)[2], 1)
            stream.receive(image_pb2.Image())
            with self.assertRaisesRegex(ValueError, "missing_image_timestamp"):
                stream.snapshot(0.5)


class TrackingTest(unittest.TestCase):
    def test_repeated_exposures_cannot_manufacture_template_confirmation(self):
        tracker = reader.TargetTracker(auto_lock=False)
        image = np.full((240, 320, 3), 190, np.uint8)
        image[80:160, 130:150] = 40
        image[110:130, 100:180] = 55
        tracker.lock_target(image, (100, 80, 80, 80), 0)
        tracker.scale = 2.
        for _ in range(10):
            tracker.upgrade_appearance(image, .3, .95, 0, True)
        self.assertEqual(tracker.template_updates, 0)
        self.assertEqual(len(tracker.appearance_votes), 1)
        self.assertEqual(tracker.target_id, 1)

    def test_optional_flow_error_falls_back_without_exiting_or_faking_lock(self):
        image = np.full((240, 320, 3), 190, np.uint8)
        image[80:150, 145:165] = 40
        image[104:124, 110:200] = 55
        tracker = reader.TargetTracker(auto_lock=False)
        tracker.update(image, 0, (100, 70, 110, 90))
        with patch.object(reader.cv2, 'calcOpticalFlowPyrLK', side_effect=cv2.error('flow unavailable')):
            result = tracker.update(image, 1/30)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['flow_support_points'], 0)
        self.assertEqual(result['target_id'], 1)

    def test_approach_with_perspective_change_keeps_identity_and_relearns_real_detail(self):
        # Unlike scaling one immutable tiny patch, render newly resolved detail
        # and changing aspect ratio. This exposes starvation of .88-only updates.
        base = np.full((100, 110, 3), 190, np.uint8)
        cv2.line(base, (22, 20), (88, 80), (40, 45, 50), 8)
        cv2.line(base, (22, 80), (88, 20), (40, 45, 50), 8)
        for point in ((22, 20), (88, 80), (22, 80), (88, 20)):
            cv2.circle(base, point, 10, (60, 65, 70), -1)
        cv2.rectangle(base, (45, 32), (65, 68), (30, 35, 40), -1)
        cv2.rectangle(base, (48, 34), (53, 42), (140, 60, 30), -1)
        for height, width in ((360, 640), (720, 1280)):
            with self.subTest(resolution=(width, height)):
                tracker = reader.TargetTracker(auto_lock=False)
                cx, cy = width//2, height//2
                for i, scale in enumerate(np.geomspace(1, 14, 180)):
                    image = np.full((height, width, 3), 190, np.uint8)
                    w = round(11*scale*(1+.7*i/179))
                    h = round(10*scale*(1-.35*i/179))
                    x, y = cx-w//2, cy-h//2
                    image[y:y+h, x:x+w] = cv2.resize(base, (w, h))
                    transform = cv2.getRotationMatrix2D((cx, cy), i*.15, 1)
                    image = cv2.warpAffine(image, transform, (width, height), borderValue=(190, 190, 190))
                    result = tracker.update(image, i/30, (cx-5, cy-5, 11, 10) if i == 0 else None)
                    self.assertTrue(result['valid'], result)
                    self.assertEqual(result['target_id'], 1)
                    np.testing.assert_allclose(result['center_px'], [cx, cy], atol=8)
                self.assertGreaterEqual(tracker.template_updates, 3)
                updates = tracker.template_updates
                missing = tracker.update(np.full_like(image, 190), 6.)
                self.assertFalse(missing['valid'])
                self.assertEqual(tracker.template_updates, updates)

    def test_moderate_scores_update_with_independent_pixels_despite_no_three_strong_frames(self):
        tracker = reader.TargetTracker(auto_lock=False)
        image = np.full((240, 320, 3), 190, np.uint8)
        image[80:160, 130:150] = 40
        image[110:130, 100:180] = 55
        tracker.lock_target(image, (100, 80, 80, 80), 0)
        tracker.scale = 2.
        for i, score in enumerate((.831, .84, .83, .85), 1):
            tracker.upgrade_appearance(image, i*.05, score, 0, True)
        self.assertEqual(tracker.template_updates, 0)  # Still respect cooldown.
        tracker.upgrade_appearance(image, .3, .83, 0, True)
        self.assertEqual(tracker.template_updates, 1)
        self.assertEqual(tracker.target_id, 1)

    def test_independent_votes_expire_and_blank_pixels_cannot_teach_appearance(self):
        tracker = reader.TargetTracker(auto_lock=False)
        image = np.full((240, 320, 3), 190, np.uint8)
        image[80:160, 130:150] = 40
        image[110:130, 100:180] = 55
        tracker.lock_target(image, (100, 80, 80, 80), 0)
        tracker.scale = 2.
        for stamp in (.1, .2, .3, 1., 2.):
            tracker.upgrade_appearance(image, stamp, .83, 0, True)
        self.assertEqual(tracker.template_updates, 0)
        for i in range(6):
            tracker.upgrade_appearance(np.full_like(image, 190), 2.1+i*.03, .85, 0, True)
        self.assertEqual(tracker.template_updates, 0)

    def test_flow_proposes_true_motion_but_cannot_validate_occlusion(self):
        tracker = reader.TargetTracker(auto_lock=False)
        image = np.full((240, 320, 3), 190, np.uint8)
        image[80:150, 145:165] = 40
        image[104:124, 110:200] = 55
        image[82:93, 148:162] = (120, 60, 20)
        tracker.update(image, 0, (100, 70, 110, 90))
        transform = cv2.getRotationMatrix2D((155, 115), 5, 1.04)
        transform[:, 2] += [6, 3]
        changed = cv2.warpAffine(image, transform, (320, 240), borderValue=(190, 190, 190))
        result = tracker.update(changed, 1/30)
        self.assertTrue(result['valid'], result)
        self.assertGreaterEqual(result['flow_support_points'], 4)
        self.assertLessEqual(result['matching_candidates'], 10)
        np.testing.assert_allclose(result['center_px'], [161, 118], atol=3)
        missing = tracker.update(np.full_like(image, 190), 2/30)
        self.assertFalse(missing['valid'])
        self.assertNotIn('center_px', missing)
        self.assertEqual(missing['flow_support_points'], 0)

    def test_faster_search_still_recovers_abrupt_near_field_rotation(self):
        image = np.full((720, 1280, 3), 190, np.uint8)
        image[310:410, 625:655] = 40
        image[348:370, 585:695] = 55
        tracker = reader.TargetTracker(auto_lock=False)
        tracker.update(image, 0, (580, 305, 120, 110))
        transform = cv2.getRotationMatrix2D((640, 360), 9, 1.06)
        changed = cv2.warpAffine(image, transform, (1280, 720), borderValue=(190, 190, 190))
        result = tracker.update(changed, .1)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['target_id'], 1)
        np.testing.assert_allclose(result['center_px'], [640, 360], atol=4)

    def test_tiny_distant_roi_upgrades_through_tenfold_approach_without_relocking(self):
        patch = np.full((18, 6, 3), 190, np.uint8)
        patch[2:16, 2:4] = 40
        patch[7:11, :] = 55
        patch[3:5, 2:4] = (120, 60, 20)
        def frame(scale):
            image = np.full((720, 1280, 3), 190, np.uint8)
            width, height = round(6*scale), round(18*scale)
            x, y = 640-width//2, 360-height//2
            image[y:y+height, x:x+width] = cv2.resize(patch, (width, height))
            return image
        tracker = reader.TargetTracker(auto_lock=False)
        tracker.update(frame(1), 0, (637, 351, 6, 18))
        for i, scale in enumerate(np.geomspace(1, 10, 180), 1):
            result = tracker.update(frame(scale), i/30)
            self.assertTrue(result['valid'], result)
            self.assertEqual(result['target_id'], 1)
            np.testing.assert_allclose(result['center_px'], [640, 360], atol=4)
            self.assertLess(abs(result['bbox_xywh'][3]-18*scale), 12)
        self.assertGreaterEqual(tracker.template_updates, 2)
        updates = tracker.template_updates
        blank = np.full((720, 1280, 3), 190, np.uint8)
        self.assertFalse(tracker.update(blank, 6.1)['valid'])
        self.assertEqual(tracker.template_updates, updates)

    def test_weak_or_discontinuous_match_cannot_relearn_target(self):
        tracker = reader.TargetTracker(auto_lock=False)
        image = np.full((240, 320, 3), 190, np.uint8)
        image[80:160, 130:150] = 40
        image[110:130, 100:180] = 55
        tracker.lock_target(image, (100, 80, 80, 80), 0)
        tracker.scale = 2.
        for i in range(1, 5):
            tracker.upgrade_appearance(image, i/10, .805, 0, True)
        self.assertEqual(tracker.template_updates, 0)
        for i in range(5, 9):
            tracker.upgrade_appearance(image, i/10, .95, 100, True)
        self.assertEqual(tracker.template_updates, 0)
        for i in range(9, 13):
            tracker.upgrade_appearance(image, i/10, .95, 0, False)
        self.assertEqual(tracker.template_updates, 0)
        for i in range(13, 16):
            tracker.upgrade_appearance(image, i/10, .95, 0, True)
        self.assertEqual(tracker.template_updates, 1)
        self.assertEqual(tracker.target_id, 1)
        self.assertEqual(tracker.next_id, 2)

    def setUp(self):
        self.texture = np.random.default_rng(42).integers(80, 256, (24, 24, 3), dtype=np.uint8)
        self.tracker = reader.TargetTracker()

    def frame(self, x=None, y=50):
        image = np.zeros((160, 240, 3), dtype=np.uint8)
        if x is not None:
            image[y:y+24, x:x+24] = self.texture
        return image

    def test_motion_warmup_detection_and_auto_lock(self):
        for index in range(12):
            result = self.tracker.update(self.frame(), index/10)
            self.assertFalse(result["valid"])
            self.assertEqual(result["candidates_xywh"], [])
        result = self.tracker.update(self.frame(40), 1.2)
        self.assertEqual(result["state"], "locked")
        self.assertEqual(result["bbox_xywh"], [40, 50, 24, 24])
        self.assertEqual(result["target_id"], 1)
        for index in range(1, 6):
            result = self.tracker.update(self.frame(40+index*5), 1.2+index/10)
            self.assertEqual(result["bbox_xywh"], [40+index*5, 50, 24, 24])
            self.assertEqual(result["state"], "tracking")

    def test_manual_lock_tracks_moving_then_stationary_target(self):
        self.tracker.update(self.frame(40), 0, (40, 50, 24, 24))
        for index in range(1, 30):
            x = min(40+index*4, 100)
            result = self.tracker.update(self.frame(x), index/10)
            self.assertTrue(result["valid"])
            self.assertEqual(result["target_id"], 1)
            self.assertEqual(result["bbox_xywh"], [x, 50, 24, 24])
            np.testing.assert_allclose(result["center_error_normalized"],
                                       [2*(x+12)/240-1, 2*62/160-1])

    def test_occlusion_search_recovery_and_permanent_loss(self):
        self.tracker.update(self.frame(40), 0, (40, 50, 24, 24))
        missing = self.tracker.update(self.frame(), 0.5)
        self.assertEqual(missing["state"], "searching")
        self.assertFalse(missing["valid"])
        self.assertIsNone(missing["bbox_xywh"])
        self.assertNotIn("center_px", missing)
        recovered = self.tracker.update(self.frame(70), 1)
        self.assertEqual(recovered["state"], "tracking")
        self.assertEqual(recovered["target_id"], 1)
        self.assertEqual(recovered["bbox_xywh"], [70, 50, 24, 24])
        self.assertEqual(self.tracker.update(self.frame(), 3.1)["state"], "lost")
        # Even a perfect match or a new motion proposal cannot silently replace a lost lock.
        lost = self.tracker.update(self.frame(80), 3.2)
        self.assertEqual(lost["state"], "lost")
        self.assertFalse(lost["valid"])
        relocked = self.tracker.update(self.frame(80), 3.3, (80, 50, 24, 24))
        self.assertEqual(relocked["target_id"], 2)
        self.assertEqual(relocked["state"], "locked")

    def test_different_appearance_is_not_accepted(self):
        self.tracker.update(self.frame(40), 0, (40, 50, 24, 24))
        image = self.frame()
        image[50:74, 60:84] = (0, 0, 255)
        result = self.tracker.update(image, 0.1)
        self.assertEqual(result["state"], "searching")
        self.assertFalse(result["valid"])

    def test_manual_mode_reset_and_resolution_change(self):
        self.tracker.auto_lock = False
        for index in range(12):
            self.tracker.update(self.frame(), index/10)
        result = self.tracker.update(self.frame(40), 1.2)
        self.assertTrue(result["candidates_xywh"])
        self.assertFalse(result["valid"])
        self.tracker.update(self.frame(40), 1.3, (40, 50, 24, 24))
        changed = self.tracker.update(np.zeros((100, 100, 3), dtype=np.uint8), 1.4)
        self.assertEqual(changed["state"], "detecting")
        self.assertIsNone(changed["target_id"])
        self.assertFalse(changed["valid"])

    def test_scale_growth_and_background_distractor(self):
        # Small textured aircraft ROI on uniform sky, with a similar-colour distractor.
        patch = np.full((40, 40, 3), 190, np.uint8)
        patch[10:30, 15:25] = (35, 45, 55)
        patch[17:23, 5:35] = (45, 55, 65)
        def frame(scale):
            image = np.full((240, 320, 3), 190, np.uint8)
            size = round(40*scale)
            image[70:70+size, 110:110+size] = cv2.resize(patch, (size, size))
            image[180:200, 20:40] = (0, 0, 255)
            return image
        self.tracker.update(frame(1), 0, (110, 70, 40, 40))
        for i, scale in enumerate(np.linspace(1, 2, 45), 1):
            result = self.tracker.update(frame(scale), i/30)
            self.assertTrue(result["valid"], result)
            self.assertEqual(result["target_id"], 1)
            self.assertLess(abs(result["bbox_xywh"][2]-40*scale), 6)
        blank = np.full((240, 320, 3), 190, np.uint8)
        self.assertFalse(self.tracker.update(blank, 1.6)["valid"])

    def test_rotation_keeps_selected_identity_and_center(self):
        initial = np.full((240, 320, 3), 190, np.uint8)
        initial[80:100, 122:138] = (20, 40, 60)
        initial[87:93, 112:148] = (40, 60, 80)
        initial[78:82, 123:128] = (200, 80, 20)
        self.tracker.update(initial, 0, (110, 70, 40, 40))
        for i, angle in enumerate(range(2, 62, 2), 1):
            matrix = cv2.getRotationMatrix2D((129.5, 89.5), angle, 1)
            image = cv2.warpAffine(initial, matrix, (320, 240), borderValue=(190, 190, 190))
            result = self.tracker.update(image, i/30)
            self.assertTrue(result["valid"], result)
            self.assertEqual(result["target_id"], 1)
            np.testing.assert_allclose(result["center_px"], [130, 90], atol=5)

    def test_large_sky_roi_cannot_lock_empty_background(self):
        image = np.full((240, 320, 3), 190, np.uint8)
        image[110:115, 150:170] = 30
        self.tracker.update(image, 0, (100, 60, 120, 120))
        empty = np.full_like(image, 190)
        result = self.tracker.update(empty, 1/30)
        self.assertFalse(result['valid'])
        self.assertEqual(result['state'], 'searching')

    def test_invalid_roi_and_black_template(self):
        for roi in [(-1, 0, 24, 24), (230, 0, 24, 24), (0, 0, 2, 2)]:
            with self.subTest(roi=roi), self.assertRaisesRegex(ValueError, "out_of_image"):
                self.tracker.update(self.frame(), 0, roi)
        with self.assertRaisesRegex(ValueError, "no_appearance"):
            self.tracker.update(self.frame(), 0, (40, 50, 24, 24))

    def test_near_field_matching_is_bounded_and_reports_original_coordinates(self):
        patch_image = np.full((640, 460, 3), 190, np.uint8)
        patch_image[80:560, 200:260] = (30, 45, 60)
        patch_image[280:350, 40:420] = (40, 55, 75)
        patch_image[120:170, 210:250] = (150, 80, 20)

        def frame(x, y):
            image = np.full((720, 1280, 3), 190, np.uint8)
            image[y:y+640, x:x+460] = patch_image
            return image

        self.tracker.update(frame(380, 35), 0, (380, 35, 460, 640))
        original_match = cv2.matchTemplate
        sizes = []

        def bounded_match(search, template, *args, **kwargs):
            sizes.append((search.shape, template.shape))
            return original_match(search, template, *args, **kwargs)

        with patch.object(reader.cv2, 'matchTemplate', side_effect=bounded_match):
            for i in range(1, 10):
                result = self.tracker.update(frame(380+i*3, 35+i), i/30)
                self.assertTrue(result['valid'], result)
                np.testing.assert_allclose(result['center_px'], [610+i*3, 355+i], atol=2)
                self.assertEqual(result['bbox_xywh'][2:], [460, 640])
        self.assertTrue(sizes)
        self.assertTrue(all(max(search) <= 192 and max(template) <= 192
                            for search, template in sizes))
        self.assertFalse(self.tracker.update(np.full((720, 1280, 3), 190, np.uint8), 0.4)['valid'])

    def test_large_target_scale_and_rotation_preserve_identity(self):
        image = np.full((720, 1280, 3), 190, np.uint8)
        image[180:520, 605:675] = (30, 45, 60)
        image[300:370, 470:810] = (45, 55, 75)
        image[230:260, 610:640] = (150, 80, 20)
        self.tracker.update(image, 0, (440, 110, 400, 500))
        for i in range(1, 13):
            transform = cv2.getRotationMatrix2D((639.5, 359.5), i*2, 1+i*.012)
            moved = cv2.warpAffine(image, transform, (1280, 720), borderValue=(190, 190, 190))
            result = self.tracker.update(moved, i/30)
            self.assertTrue(result['valid'], result)
            self.assertEqual(result['target_id'], 1)
            np.testing.assert_allclose(result['center_px'], [640, 360], atol=5)

    def test_static_aircraft_with_subpixel_jitter_does_not_drift_scale(self):
        image = np.full((240, 320, 3), 190, np.uint8)
        image[90:124, 151:169] = 40
        image[101:110, 135:184] = 55
        self.tracker.update(image, 0, (135, 89, 49, 36))
        for i in range(1, 121):
            transform = np.float32([[1, 0, np.sin(i*.2)*.6], [0, 1, np.cos(i*.11)*.6]])
            moved = cv2.warpAffine(image, transform, (320, 240), borderValue=(190, 190, 190))
            result = self.tracker.update(moved, i/30)
            self.assertTrue(result['valid'], result)
            self.assertGreater(result['scale'], .90)
            self.assertLess(result['scale'], 1.10)
            np.testing.assert_allclose(result['center_px'], [159.5, 107], atol=2)

    def test_uniform_larger_blob_cannot_replace_selected_aircraft(self):
        image = np.full((240, 320, 3), 190, np.uint8)
        image[90:124, 151:169] = 40
        image[101:110, 135:184] = 55
        self.tracker.update(image, 0, (135, 89, 49, 36))
        wrong = np.full_like(image, 190)
        wrong[70:142, 110:208] = 45
        before = self.tracker.box
        for i in range(1, 6):
            result = self.tracker.update(wrong, i/30)
            self.assertFalse(result['valid'], result)
            self.assertEqual(self.tracker.box, before)
            self.assertEqual(self.tracker.scale, 1.)

    def test_true_target_shrinking_is_still_tracked(self):
        patch = np.full((40, 40, 3), 190, np.uint8)
        patch[10:30, 15:25] = (35, 45, 55)
        patch[17:23, 5:35] = (45, 55, 65)
        def frame(scale):
            image = np.full((240, 320, 3), 190, np.uint8)
            size = round(40*scale)
            image[70:70+size, 110:110+size] = cv2.resize(patch, (size, size))
            return image
        self.tracker.update(frame(1), 0, (110, 70, 40, 40))
        for i, scale in enumerate(np.linspace(1, .5, 45), 1):
            result = self.tracker.update(frame(scale), i/30)
            self.assertTrue(result['valid'], result)
            self.assertLess(abs(result['bbox_xywh'][2]-40*scale), 4)

    def test_motion_prediction_tracks_fast_target_without_publishing_predictions(self):
        def frame(x=None):
            image = np.zeros((160, 640, 3), np.uint8)
            if x is not None:
                image[50:74, x:x+24] = self.texture
            return image
        self.tracker.update(frame(20), 0, (20, 50, 24, 24))
        for i, x in enumerate((40, 70, 120, 190, 260), 1):
            result = self.tracker.update(frame(x), i/30)
            self.assertTrue(result['valid'], result)
            np.testing.assert_allclose(result['center_px'], [x+12, 62], atol=1)
        missing = self.tracker.update(frame(), 6/30)
        self.assertFalse(missing['valid'])
        self.assertNotIn('center_px', missing)

    def test_distant_recovery_needs_two_real_matches_of_the_same_identity(self):
        self.tracker.update(self.frame(40), 0, (40, 50, 24, 24))
        self.assertFalse(self.tracker.update(self.frame(), 1/30)['valid'])
        first = self.tracker.update(self.frame(110), .2)
        self.assertFalse(first['valid'], first)
        self.assertTrue(first['recovery_pending'], first)
        self.assertNotIn('center_px', first)
        second = self.tracker.update(self.frame(110), .2+1/30)
        self.assertTrue(second['valid'], second)
        self.assertEqual(second['target_id'], 1)
        np.testing.assert_allclose(second['center_px'], [122, 62], atol=1)

    def test_blank_frame_does_not_change_velocity_or_identity(self):
        self.tracker.update(self.frame(40), 0, (40, 50, 24, 24))
        self.tracker.update(self.frame(50), 1/30)
        previous = self.tracker.velocity.copy()
        result = self.tracker.update(self.frame(), 2/30)
        self.assertFalse(result['valid'])
        self.assertEqual(result['target_id'], 1)
        np.testing.assert_equal(self.tracker.velocity, previous)

    def test_identical_distractor_does_not_replace_expected_position(self):
        self.tracker.update(self.frame(100), 0, (100, 50, 24, 24))
        image = self.frame(100)
        image[50:74, 60:84] = self.texture
        result = self.tracker.update(image, 1/30)
        self.assertTrue(result['valid'], result)
        np.testing.assert_allclose(result['center_px'], [112, 62], atol=1)

    def test_miss_between_recovery_candidates_resets_confirmation(self):
        self.tracker.update(self.frame(40), 0, (40, 50, 24, 24))
        self.tracker.update(self.frame(), 1/30)
        first = self.tracker.update(self.frame(110), .2)
        self.assertTrue(first['recovery_pending'])
        missing = self.tracker.update(self.frame(), .2+1/30)
        self.assertFalse(missing['recovery_pending'])
        next_match = self.tracker.update(self.frame(110), .2+2/30)
        self.assertFalse(next_match['valid'])
        self.assertTrue(next_match['recovery_pending'])
        confirmed = self.tracker.update(self.frame(110), .3)
        self.assertTrue(confirmed['valid'], confirmed)

    def test_sim_gimbal_rate_argument_is_finite_and_bounded(self):
        self.assertEqual(reader.gimbal_rate('1'), 1.)
        for value in ('0', '-1', '2.1', 'nan', 'inf'):
            with self.assertRaises(reader.argparse.ArgumentTypeError):
                reader.gimbal_rate(value)


class CLITest(unittest.TestCase):
    def run_tracking_stream(self, interrupted=False):
        texture = np.random.default_rng(42).integers(80, 256, (24, 24, 3), dtype=np.uint8)

        def message(x, stamp):
            image = np.zeros((160, 240, 3), dtype=np.uint8)
            image[50:74, x:x+24] = texture
            result = image_pb2.Image(width=240, height=160, step=720,
                                     pixel_format_type=image_pb2.BGR_INT8, data=image.tobytes())
            result.header.stamp.sec = stamp
            return result

        class ReadOnlyNode:
            def unsubscribe(self, topic):
                return True

            def subscribe(self, message_type, topic, callback):
                return True

        first = message(40, 10)
        second = message(50, 11)
        snapshots = [(first, 1, 0, 0), (second, 2, 0, 0)]
        if interrupted:
            snapshots += [ValueError("stale_image"), (message(60, 12), 3, 0, 0)]
        else:
            snapshots += [(second, 3, 0, 0), (message(60, 0), 4, 1, 0)]
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        arguments = ["reader", "--topic", "/test/image", "--headless", "--manual-lock",
                     "--roi", "40,50,24,24", "--duration", "0.35", "--rate", "10"]
        output = io.StringIO()
        with patch("gz.transport13.Node", ReadOnlyNode), \
                patch.object(reader.ImageStream, "snapshot", side_effect=snapshots), \
                patch.object(reader.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(reader.time, "sleep", side_effect=sleep), \
                patch.object(reader.sys, "argv", arguments), \
                patch.object(reader.sys, "stdout", output), \
                patch.object(reader.sys, "stderr", io.StringIO()):
            self.assertEqual(reader.main(), 0)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_tracking_cli_duplicate_stamp_and_epoch_reset(self):
        records = self.run_tracking_stream()
        frames = [record for record in records if record["type"] == "frame"]
        self.assertEqual(frames[0]["target"]["state"], "locked")
        self.assertEqual(frames[1]["target"]["bbox_xywh"], [50, 50, 24, 24])
        self.assertEqual(frames[1]["target"], frames[2]["target"])
        self.assertEqual(frames[3]["sim_epoch"], 1)
        self.assertEqual(frames[3]["target"]["state"], "detecting")
        self.assertFalse(frames[3]["target"]["valid"])

    def test_tracking_cli_preserves_identity_after_brief_stream_interruption(self):
        records = self.run_tracking_stream(interrupted=True)
        states = [record["state"] for record in records if record["type"] == "stream_state"]
        self.assertIn("stale_image", states)
        frames = [record for record in records if record["type"] == "frame"]
        self.assertEqual(frames[-1]["target"]["state"], "tracking")
        self.assertTrue(frames[-1]["target"]["valid"])
        self.assertEqual(frames[-1]["target"]["target_id"], frames[0]["target"]["target_id"])

    def test_pose_failure_does_not_clear_image_lock(self):
        texture = np.random.default_rng(17).integers(80, 256, (24, 24, 3), dtype=np.uint8)
        snapshots = []
        for index in range(3):
            image = np.zeros((160, 240, 3), dtype=np.uint8)
            image[50:74, 40+index:64+index] = texture
            message = image_pb2.Image(width=240, height=160, step=720,
                pixel_format_type=image_pb2.BGR_INT8, data=image.tobytes())
            message.header.stamp.sec = 10
            message.header.stamp.nsec = index*33_000_000
            snapshots.append((message, index+1, 0, 0))
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        output = io.StringIO()
        arguments = ["reader", "--topic", "/test/image", "--headless", "--manual-lock",
            "--roi", "40,50,24,24", "--duration", "0.1", "--rate", "30", "--dyt-model", "test"]
        with patch("gz.transport13.Node"), patch('gz_dyt_bridge.DytBridge') as bridge,                 patch.object(reader.ImageStream, 'snapshot', side_effect=snapshots),                 patch.object(reader.time, 'monotonic', side_effect=lambda: clock[0]),                 patch.object(reader.time, 'sleep', side_effect=sleep),                 patch.object(reader.sys, 'argv', arguments),                 patch.object(reader.sys, 'stdout', output),                 patch.object(reader.sys, 'stderr', io.StringIO()):
            bridge.return_value.update.side_effect = [None, ValueError('dyt_image_pose_time_mismatch'), None]
            bridge.return_value.invalidate.return_value = {'loss_grace_active': False, 'measurement_published': True}
            self.assertEqual(reader.main(), 0)
            bridge.return_value.invalidate.assert_any_call(force=True)
            bridge.return_value.invalidate.assert_any_call(stamp_ns=10_033_000_000)
        frames = [json.loads(line) for line in output.getvalue().splitlines()
                  if json.loads(line)['type'] == 'frame']
        self.assertEqual([frame['target']['target_id'] for frame in frames], [1, 1, 1])
        self.assertEqual([frame['dyt_valid'] for frame in frames], [True, False, True])

    def run_reader(self, directory):
        class ReadOnlyNode:
            def unsubscribe(self, topic):
                return True

            def subscribe(self, message_type, topic, callback):
                if topic != "/test/image":
                    raise AssertionError("wrong subscription")
                callback(image_message())
                return True

        arguments = ["reader", "--topic", "/test/image", "--headless", "--record-dir", str(directory),
                     "--duration", "0.02", "--rate", "100"]
        output = io.StringIO()
        with patch("gz.transport13.Node", ReadOnlyNode), patch.object(reader.sys, "argv", arguments), \
                patch.object(reader.sys, "stdout", output), patch.object(reader.sys, "stderr", io.StringIO()), \
                patch.object(reader.cv2, "namedWindow", side_effect=AssertionError("headless opened window")):
            code = reader.main()
        return code, output.getvalue()

    def test_save_pixels_metadata_and_no_duplicate_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/"capture"
            code, output = self.run_reader(directory)
            self.assertEqual(code, 0)
            records = [json.loads(line) for line in output.splitlines()]
            frames = [record for record in records if record["type"] == "frame"]
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0]["stamp_ns"], 12_000_000_345)
            np.testing.assert_array_equal(cv2.imread(str(directory/frames[0]["png"])),
                                          reader.decode_image(image_message()))
            self.assertEqual((directory/"frames.jsonl").read_text(), output)
            original = (directory/"frames.jsonl").read_text()
            self.assertEqual(self.run_reader(directory)[0], 1)
            self.assertEqual((directory/"frames.jsonl").read_text(), original)


if __name__ == "__main__":
    unittest.main()
