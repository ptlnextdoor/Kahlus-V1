import dataclasses
import unittest

from neurotwin.researchdock.schemas import (
    ResearchDockSensorPacket,
    ResearchDockSelfReport,
    ResearchDockSession,
    ResearchDockTaskEvent,
    ResearchDockTrial,
)
from neurotwin.researchdock.tasks import researchdock_task_templates


class ResearchDockSchemaTests(unittest.TestCase):
    def test_trial_and_session_creation_requires_no_pii(self):
        sensor = ResearchDockSensorPacket(
            timestamp=1.0,
            pupil_diameter=3.2,
            gaze_x=0.1,
            gaze_y=-0.2,
            ppg_value=0.8,
            hrv_proxy=62.0,
        )
        event = ResearchDockTaskEvent(
            timestamp=1.0,
            task_name="reward_anticipation",
            event_type="cue",
            reward_condition="reward",
            effort_level=0.25,
            reaction_time_ms=410.0,
            accuracy=1.0,
        )
        report = ResearchDockSelfReport(timestamp=1.2, valence=0.4, arousal=0.3, motivation=0.7)
        trial = ResearchDockTrial(
            participant_id_hash="sha256:abc",
            session_id="ses-001",
            timestamp=1.0,
            task_name="reward_anticipation",
            event_type="cue",
            reward_condition="reward",
            effort_level=0.25,
            reaction_time_ms=410.0,
            accuracy=1.0,
            sensor_packet=sensor,
            task_event=event,
            self_report=report,
        )
        session = ResearchDockSession(
            participant_id_hash="sha256:abc",
            session_id="ses-001",
            trials=(trial,),
        )

        self.assertEqual(session.trials[0].participant_id_hash, "sha256:abc")
        for cls in (
            ResearchDockSensorPacket,
            ResearchDockSelfReport,
            ResearchDockSession,
            ResearchDockTaskEvent,
            ResearchDockTrial,
        ):
            field_names = {field.name for field in dataclasses.fields(cls)}
            self.assertIn("quality_flags", field_names)
            self.assertNotIn("name", field_names)
            self.assertNotIn("email", field_names)
            self.assertNotIn("date_of_birth", field_names)

    def test_task_templates_are_safe_and_complete(self):
        templates = researchdock_task_templates()
        template_ids = {template.task_name for template in templates}

        self.assertEqual(
            template_ids,
            {
                "reward_anticipation",
                "probabilistic_reward_learning",
                "effort_for_reward",
                "mild_frustration",
                "recovery_rest",
                "visual_attention",
            },
        )
        joined = " ".join(template.description.lower() for template in templates)
        self.assertNotIn("trauma exposure", joined)
        self.assertNotIn("treatment", joined)
        self.assertNotIn("diagnosis", joined)


if __name__ == "__main__":
    unittest.main()
