import IPython.display
import ipywidgets as ipw
import nwbwidgets.misc
import pynwb

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


class PSTHWidget(nwbwidgets.misc.PSTHWidget):
    def __init__(
        self,
        input_data: pynwb.misc.Units,
        trials: pynwb.epoch.TimeIntervals = None,
        unit_index=0,
        unit_controller=None,
        ntt=1000,
    ) -> None:
        super().__init__(
            input_data=input_data,
            trials=trials,
            unit_index=unit_index,
            unit_controller=unit_controller,
            ntt=ntt,
        )
        self.start_ft.value = -0.1
        self.end_ft.value = 0.4
        self.psth_type_radio.value = "gaussian"
        self.gaussian_sd_ft.value = 0.005
        if "stim_start_time" in self.trial_event_controller.options:
            self.trial_event_controller.value = ["stim_start_time"]
        else:
            self.trial_event_controller.value = ["start_time"]
        if "stim_name" in self.gas.group_dd.options:
            self.gas.group_dd.value = "stim_name"

    def update(self, *args, **kwargs):
        kwargs.setdefault("figsize", (6, 7))
        fig = super(self.__class__, self).update(*args, **kwargs)
        fig.axes[1].set_xticks(sorted([*fig.axes[1].get_xlim(), 0]))
        return fig

    __doc__ = nwbwidgets.misc.PSTHWidget.__doc__
    __init__.__doc__ = nwbwidgets.misc.PSTHWidget.__init__.__doc__
