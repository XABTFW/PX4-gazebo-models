"""Elastic-net geometry, conservation, trigger and asset tests (no live flight)."""
import unittest
import xml.etree.ElementTree as ET
import numpy as np

from gz_net_physics import (NetConfig, grid, rope_forces, sphere_contact, wrapping_nodes,
                            closest_triangles, model_sdf, launcher_sdf, valid_name, capture_confirmed,
                            dense_net_segments)
from gz_net_capture import demo_world


class NetPhysicsTest(unittest.TestCase):
    def test_rendered_net_is_dense_and_tracks_real_node_deformation(self):
        positions = self.directions*self.config.width/2
        segments = dense_net_segments(positions, 7, 5.)
        self.assertGreater(len(segments), 4*len(self.visible))
        np.testing.assert_allclose(np.ptp(segments.reshape(-1, 3), axis=0), [0, 5, 5])
        translation = np.array([10., 20, 30])
        np.testing.assert_allclose(dense_net_segments(positions+translation, 7, 5.), segments+translation)
        folded = positions.copy()
        folded[:, 0] = folded[:, 1]**2*.1
        bent = dense_net_segments(folded, 7, 5.)
        self.assertEqual(bent.shape, segments.shape)
        self.assertGreater(float(np.max(bent[:, :, 0])), .5)
        np.testing.assert_allclose(positions[:, 0], 0)  # Renderer cannot mutate physical state.

    def test_net_perimeter_is_five_metres_with_dense_body_contact_samples(self):
        self.assertEqual(self.config.width, 5.)
        self.assertEqual(self.config.spread_speed, 5.)
        deployed = self.directions*self.config.width/2
        np.testing.assert_allclose(np.ptp(deployed, axis=0), [0, 5, 5])
        old = NetConfig(width=2., spread_speed=2.)
        old_directions = grid(old)[1]
        core = [row*7+col for row in range(1, 6) for col in range(1, 6)]
        np.testing.assert_allclose(deployed[core], old_directions[core])
        np.testing.assert_allclose(self.directions[core]*self.config.spread_speed,
                                   old_directions[core]*old.spread_speed)

    def test_release_proximity_or_single_touch_cannot_confirm_body_capture(self):
        hold = self.config.capture_hold
        self.assertFalse(capture_confirmed(0, 0, 0, 0, hold))
        self.assertFalse(capture_confirmed(10, 0, 1, 0, hold))
        self.assertFalse(capture_confirmed(10, 4, .004, .004, hold))
        self.assertFalse(capture_confirmed(0, 4, 1, 1, hold))
        self.assertFalse(capture_confirmed(10, 4, 0, 1, hold))
        self.assertTrue(capture_confirmed(10, 4, hold, hold, hold))

    def test_default_release_distance_is_five_metres(self):
        root = ET.fromstring(model_sdf('eco_net', NetConfig(), 'observer', 'target'))
        self.assertEqual(float(root.find('.//plugin/trigger_distance').text), 5.)

    def setUp(self):
        self.config = NetConfig().validate()
        self.folded, self.directions, self.edges, self.rest, self.faces, self.visible, self.masses = grid(self.config)

    def test_mesh_topology_and_mass(self):
        self.assertEqual(self.folded.shape, (49, 3))
        self.assertEqual(len(self.faces), 72)
        self.assertEqual(len(self.visible), 84)
        self.assertAlmostEqual(sum(self.masses), .25)
        self.assertTrue(np.all(self.rest > 0))
        self.assertTrue(np.all(self.masses > 0))

    def test_slack_ropes_never_push_and_rest_grid_has_zero_force(self):
        for positions in (self.folded, self.directions*self.config.width/2):
            force, _ = rope_forces(positions, np.zeros_like(positions), self.edges, self.rest, 12, .04)
            np.testing.assert_allclose(force, 0, atol=1e-12)

    def test_stretched_internal_forces_conserve_momentum_and_torque(self):
        positions = self.directions*self.config.width
        force, tension = rope_forces(positions, np.zeros_like(positions), self.edges, self.rest, 12, .04)
        self.assertGreater(tension, 0)
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(np.cross(positions, force).sum(axis=0), 0, atol=1e-10)

    def test_damping_does_not_add_energy_to_stretched_net(self):
        positions = self.directions*self.config.width
        velocities = positions.copy()
        elastic, _ = rope_forces(positions, velocities, self.edges, self.rest, 12, 0)
        damped, _ = rope_forces(positions, velocities, self.edges, self.rest, 12, .1)
        self.assertLess(float(np.sum((damped-elastic)*velocities)), 0)

    def test_triangle_projection_interior_edge_and_degenerate(self):
        triangles = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0]]*3])
        closest, weights = closest_triangles(np.array([.2, .2, 1]), triangles)
        np.testing.assert_allclose(closest[0], [.2, .2, 0])
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        self.assertTrue(np.all(weights >= -1e-10))
        self.assertTrue(np.all(np.isfinite(closest)))

    def test_sphere_contact_reaction_conserves_momentum(self):
        positions = self.directions
        force, reaction, torque, contacts = sphere_contact(positions, np.zeros_like(positions), self.faces,
                    np.array([.1, 0, 0]), np.zeros(3), .35, 15, .12)
        self.assertGreater(contacts, 0)
        np.testing.assert_allclose(force.sum(axis=0)+reaction, 0, atol=1e-10)
        np.testing.assert_allclose(np.cross(positions, force).sum(axis=0)+
                                   np.cross([.1, 0, 0], reaction)+torque, 0, atol=1e-10)

    def test_no_contact_at_distance_and_finite_at_exact_plane(self):
        for center in (np.array([2., 0, 0]), np.zeros(3)):
            force, reaction, _, contacts = sphere_contact(self.directions, self.directions+np.array([1., 0, 0]),
                            self.faces, center, np.zeros(3), .35, 15, .12)
            self.assertTrue(np.all(np.isfinite(force)))
            self.assertTrue(np.all(np.isfinite(reaction)))
            if center[0] == 2:
                self.assertEqual(contacts, 0)

    def test_touching_one_side_cannot_be_called_capture(self):
        positions = np.array([[-.2, .15, .15], [-.2, -.15, .15], [-.2, .15, -.15], [-.2, -.15, -.15]])
        self.assertEqual(len(wrapping_nodes(positions, np.zeros(3), .35, np.array([1., 0, 0]))), 0)
        positions[0:2, 0] = .2
        self.assertEqual(len(wrapping_nodes(positions, np.zeros(3), .35, np.array([1., 0, 0]))), 4)

    def test_invalid_parameters_and_names_are_rejected(self):
        for config in (NetConfig(resolution=4), NetConfig(mass=-1), NetConfig(launch_speed=float('nan')),
                       NetConfig(target_radius=.01), NetConfig(folded_width=10), NetConfig(spread_speed=13)):
            with self.assertRaises(ValueError): config.validate()
        for name in ('../x', 'x;rm', 'x/y', '<bad>', ''):
            with self.assertRaises(ValueError): valid_name(name)

    def test_generated_model_is_dynamic_and_has_real_particle_inertia(self):
        root = ET.fromstring(model_sdf('eco_net', self.config, 'observer', 'target'))
        model = root.find('model')
        self.assertEqual(len(model.findall('link')), 49)
        self.assertIsNone(model.find('static'))
        self.assertEqual(model.find('plugin/auto_fire').text, 'false')
        self.assertEqual(model.find('plugin/require_lock').text, 'true')
        self.assertEqual(model.find('plugin/demo').text, 'false')
        self.assertEqual(model.find('plugin/drop_on_capture').text, 'false')
        offset = list(map(float, model.find('plugin/offset').text.split()))
        self.assertGreater(np.linalg.norm(offset[:2]), .105+.024)  # Do not enclose the original camera.
        self.assertAlmostEqual(sum(float(link.find('inertial/mass').text) for link in model.findall('link')), .25)
        ET.fromstring(launcher_sdf('eco_net_launcher'))

    def test_passive_litter_proxy_requires_explicit_option(self):
        root = ET.fromstring(model_sdf('eco_net', self.config, 'observer', 'x500_2', drop_on_capture=True))
        self.assertEqual(root.find('model/plugin/drop_on_capture').text, 'true')
        self.assertEqual(root.find('model/plugin/target').text, 'x500_2')

    def test_invisible_net_keeps_identical_physics_and_capture_configuration(self):
        hidden = ET.fromstring(model_sdf('eco_net', self.config, 'observer', 'x500_2', drop_on_capture=True))
        shown = ET.fromstring(model_sdf('eco_net', self.config, 'observer', 'x500_2', drop_on_capture=True, visible=True))
        self.assertEqual(len(hidden.findall('.//visual')), 0)
        self.assertEqual(len(shown.findall('.//visual')), 49)
        for link in shown.findall('model/link'):
            for visual in list(link.findall('visual')):
                link.remove(visual)
        self.assertEqual(ET.tostring(hidden), ET.tostring(shown))
        self.assertEqual(len(hidden.findall('model/link')), 49)

    def test_hidden_launcher_has_no_visuals_and_demo_remains_visible(self):
        hidden = ET.fromstring(launcher_sdf('eco_net_launcher'))
        shown = ET.fromstring(launcher_sdf('eco_net_launcher', visible=True))
        self.assertEqual(len(hidden.findall('.//visual')), 0)
        self.assertEqual(len(shown.findall('.//visual')), 2)
        for visual in list(shown.find('model/link').findall('visual')):
            shown.find('model/link').remove(visual)
        self.assertEqual(ET.tostring(hidden), ET.tostring(shown))
        demo = ET.fromstring(demo_world(self.config))
        self.assertEqual(len(demo.findall('world/model[@name="eco_net"]/link/visual')), 49)

    def test_demo_is_separate_and_not_px4(self):
        root = ET.fromstring(demo_world(self.config))
        self.assertEqual(root.find('world').get('name'), 'net_demo')
        self.assertNotIn('x500', demo_world(self.config))
        self.assertEqual(root.find('world/model[@name="eco_net"]/plugin/demo').text, 'true')


if __name__ == '__main__':
    unittest.main()
