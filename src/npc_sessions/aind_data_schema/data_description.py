import logging

import aind_data_schema.components.identifiers
import aind_data_schema.core.data_description
import aind_data_schema_models.data_name_patterns
import aind_data_schema_models.modalities
import aind_data_schema_models.organizations

from npc_sessions.sessions import DynamicRoutingSession

logger = logging.getLogger(__name__)

def get_data_description_model(session: DynamicRoutingSession) -> aind_data_schema.core.data_description.DataDescription:
    """Get the Pydantic model corresponding to the 'data_description.json' for a given session."""
    subject_id = str(session.id.subject)
    creation_time = session.session_start_time
    platform = 'ecephys' if session.is_ephys else 'behavior'
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
        tags=session.keywords,
        data_summary=session.experiment_description,
    )


if __name__ == "__main__":
    # Example usage
    session = DynamicRoutingSession('814666_20251107')
    data_description = get_data_description_model(session)
    print(data_description.model_dump_json(indent=2))
    session = DynamicRoutingSession('628801_2022-09-20')
    data_description = get_data_description_model(session)
    print(data_description.model_dump_json(indent=2))
