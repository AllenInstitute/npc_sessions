import IPython.display
import ipywidgets as ipw

from npc_sessions.sessions import DynamicRoutingSession, get_sessions


def session_widget() -> DynamicRoutingSession:
    session = DynamicRoutingSession("dummy 366122 2023-01-01")
    session_select = ipw.Select(
        value=None,
        options=[
            f"{s.id.date} {s.info.project if s.info else '?'} {s.id.subject}"
            for s in get_sessions()
        ],
        disabled=False,
    )
    session_specify = ipw.Text(
        value=None,
        placeholder="Specify subject ID and date",
        disabled=False,
        continuous_update=False,
    )
    console = ipw.Output()

    @console.capture(clear_output=True)
    def update(change) -> None:
        if DynamicRoutingSession(change["new"]) != session:
            # we can't re-return the session, so we have to update it in place:
            session.__init__(change["new"])  # type: ignore[misc]
            print(
                f"Updated `session`: {session!r}\n(to make another session, re-run this cell)"
            )
        session_select.disabled = True
        session_specify.disabled = True

    session_select.observe(update, names=["value"])
    session_specify.observe(update, names=["value"])

    IPython.display.display(ipw.VBox([session_select, session_specify, console]))
    return session
