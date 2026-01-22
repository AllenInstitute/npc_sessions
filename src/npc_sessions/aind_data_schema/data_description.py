import logging

import aind_data_schema.components.identifiers
import aind_data_schema.core.data_description
import aind_data_schema_models.data_name_patterns
import aind_data_schema_models.modalities
import aind_data_schema_models.organizations

from npc_sessions.sessions import DynamicRoutingSession

logger = logging.getLogger(__name__)


def get_data_description_model(
    session: DynamicRoutingSession,
) -> aind_data_schema.core.data_description.DataDescription:
    """Get the Pydantic model corresponding to the 'data_description.json' for a given session."""
    subject_id = str(session.id.subject)
    creation_time = session.session_start_time
    platform = "ecephys" if session.is_ephys else "behavior"
    name = f"{platform}_{subject_id}_{creation_time.strftime('%Y-%m-%d_%H-%M-%S')}"

    investigators = [
        aind_data_schema.components.identifiers.Person(
            name="Shawn Olsen",
            registry=aind_data_schema.components.identifiers.Registry.ORCID,
            registry_identifier="0000-0002-9568-7057",
        ),
        # others added below depending on project
    ]

    if session.is_templeton:
        project_name = "Templeton"
        funding_source = aind_data_schema.core.data_description.Funding(
            funder=aind_data_schema_models.organizations.Organization.TWCF,
        )
        investigators.append(
            aind_data_schema.components.identifiers.Person(
                name="Ethan McBride",
                registry=aind_data_schema.components.identifiers.Registry.ORCID,
                registry_identifier="0000-0001-9489-2828",
            )
        )
    else:
        project_name = "Dynamic Routing"
        funding_source = aind_data_schema.core.data_description.Funding(
            funder=aind_data_schema_models.organizations.Organization.AI,
        )
        investigators.append(
            aind_data_schema.components.identifiers.Person(
                name="Corbett Bennett",
                registry=aind_data_schema.components.identifiers.Registry.ORCID,
                registry_identifier="0009-0001-2847-7754",
            )
        )

    modalities = []
    if session.is_ephys:
        modalities.append(aind_data_schema_models.modalities.Modality.ECEPHYS)
    if session.is_task:
        modalities.append(aind_data_schema_models.modalities.Modality.BEHAVIOR)
    if session.is_video:
        modalities.append(aind_data_schema_models.modalities.Modality.BEHAVIOR_VIDEOS)

    tags: list[str] = []
    if session.is_surface_channels:
        tags.append("surface_recording")
    if session.is_hab:
        tags.append("hab")
    if session.is_opto:
        tags.append("opto_perturbation")
    elif session.is_opto_control:
        tags.append("opto_control")
    if session.is_production:
        tags.append("production")
    else:
        tags.append("development")
    if session.is_injection_perturbation:
        tags.append("injection_perturbation")
    elif session.is_injection_control:
        tags.append("injection_control")
    if session.is_context_naive:
        tags.append("context_naive")
    if session.is_naive:
        tags.append("naive")
    if session.is_stage_5_passed:
        tags.append("stage_5_passed")
    if session.is_task and session.is_first_block_aud:
        tags.append("first_block_aud")
    elif session.is_task and session.is_first_block_vis:
        tags.append("first_block_vis")
    if session.is_task and session.is_late_autorewards:
        tags.append("late_autorewards")
    elif session.is_task:
        tags.append("early_autorewards")

    return aind_data_schema.core.data_description.DataDescription(
        name=name,
        institution=aind_data_schema_models.organizations.Organization.AIND,
        funding_source=[funding_source],
        data_level=aind_data_schema_models.data_name_patterns.DataLevel.RAW,
        group=aind_data_schema_models.data_name_patterns.Group.EPHYS,
        subject_id=subject_id,
        creation_time=creation_time,
        investigators=investigators,
        project_name=project_name,
        restrictions=None,
        modalities=modalities,
        tags=tags,
        data_summary=session.experiment_description,
    )


if __name__ == "__main__":
    # Example usage
    session = DynamicRoutingSession("814666_20251107")
    data_description = get_data_description_model(session)
    print(data_description.model_dump_json(indent=2))
    session = DynamicRoutingSession("628801_2022-09-20")
    data_description = get_data_description_model(session)
    print(data_description.model_dump_json(indent=2))
