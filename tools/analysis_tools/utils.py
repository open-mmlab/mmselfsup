from matplotlib import _api
from matplotlib import transforms as mtransforms
import matplotlib as mpl


class _DummyAxis:
    __name__ = "dummy"

    # Once the deprecation elapses, replace dataLim and viewLim by plain
    # _view_interval and _data_interval private tuples.
    dataLim = _api.deprecate_privatize_attribute(
        "3.6", alternative="get_data_interval() and set_data_interval()")
    viewLim = _api.deprecate_privatize_attribute(
        "3.6", alternative="get_view_interval() and set_view_interval()")

    def __init__(self, minpos=0):
        self._dataLim = mtransforms.Bbox.unit()
        self._viewLim = mtransforms.Bbox.unit()
        self._minpos = minpos

    def get_view_interval(self):
        return self._viewLim.intervalx

    def set_view_interval(self, vmin, vmax):
        self._viewLim.intervalx = vmin, vmax

    def get_minpos(self):
        return self._minpos

    def get_data_interval(self):
        return self._dataLim.intervalx

    def set_data_interval(self, vmin, vmax):
        self._dataLim.intervalx = vmin, vmax

    def get_tick_space(self):
        # Just use the long-standing default of nbins==9
        return 9


class TickHelper:
    axis = None

    def set_axis(self, axis):
        self.axis = axis

    def create_dummy_axis(self, **kwargs):
        if self.axis is None:
            self.axis = _DummyAxis(**kwargs)

    @_api.deprecated("3.5", alternative="`.Axis.set_view_interval`")
    def set_view_interval(self, vmin, vmax):
        self.axis.set_view_interval(vmin, vmax)

    @_api.deprecated("3.5", alternative="`.Axis.set_data_interval`")
    def set_data_interval(self, vmin, vmax):
        self.axis.set_data_interval(vmin, vmax)

    @_api.deprecated(
        "3.5",
        alternative="`.Axis.set_view_interval` and `.Axis.set_data_interval`")
    def set_bounds(self, vmin, vmax):
        self.set_view_interval(vmin, vmax)
        self.set_data_interval(vmin, vmax)


class Formatter(TickHelper):
    """
    Create a string based on a tick value and location.
    """
    # some classes want to see all the locs to help format
    # individual ones
    locs = []

    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position pos.
        ``pos=None`` indicates an unspecified location.
        """
        raise NotImplementedError('Derived must override')

    def format_ticks(self, values):
        """Return the tick labels for all the ticks at once."""
        self.set_locs(values)
        return [self(value, i) for i, value in enumerate(values)]

    def format_data(self, value):
        """
        Return the full string representation of the value with the
        position unspecified.
        """
        return self.__call__(value)

    def format_data_short(self, value):
        """
        Return a short string version of the tick value.

        Defaults to the position-independent long value.
        """
        return self.format_data(value)

    def get_offset(self):
        return ''

    def set_locs(self, locs):
        """
        Set the locations of the ticks.

        This method is called before computing the tick labels because some
        formatters need to know all tick locations to do so.
        """
        self.locs = locs

    @staticmethod
    def fix_minus(s):
        """
        Some classes may want to replace a hyphen for minus with the proper
        Unicode symbol (U+2212) for typographical correctness.  This is a
        helper method to perform such a replacement when it is enabled via
        :rc:`axes.unicode_minus`.
        """
        return (s.replace('-', '\N{MINUS SIGN}')
                if mpl.rcParams['axes.unicode_minus'] else s)

    def _set_locator(self, locator):
        """Subclasses may want to override this to set a locator."""
        pass


class FormatStrFormatter(Formatter):
    """
    Use an old-style ('%' operator) format string to format the tick.

    The format string should have a single variable format (%) in it.
    It will be applied to the value (not the position) of the tick.

    Negative numeric values will use a dash, not a Unicode minus; use mathtext
    to get a Unicode minus by wrapping the format specifier with $ (e.g.
    "$%g$").
    """

    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        """
        Return the formatted label string.

        Only the value *x* is formatted. The position is ignored.
        """
        return self.fmt % x


class ShapeBias:
    """Reference: Geirhos et al. ICLR 2019
    https://openreview.net/pdf?id=Bygh9j09KX
    """
    num_input_models = 1

    def __init__(self):
        super().__init__()
        self.plotting_name = "shape-bias"

    @staticmethod
    def _check_dataframe(df):
        assert len(df) > 0, "empty dataframe"

    def analysis(self, df):

        self._check_dataframe(df)

        df = df.copy()
        df["correct_texture"] = df["imagename"].apply(
            self.get_texture_category)
        df["correct_shape"] = df["category"]

        # remove those rows where shape = texture, i.e. no cue conflict present
        df2 = df.loc[df.correct_shape != df.correct_texture]
        fraction_correct_shape = len(
            df2.loc[df2.object_response == df2.correct_shape]) / len(df)
        fraction_correct_texture = len(
            df2.loc[df2.object_response == df2.correct_texture]) / len(df)
        shape_bias = fraction_correct_shape / (
            fraction_correct_shape + fraction_correct_texture)

        result_dict = {
            "fraction-correct-shape": fraction_correct_shape,
            "fraction-correct-texture": fraction_correct_texture,
            "shape-bias": shape_bias
        }
        return result_dict

    def get_result_df(self):
        pass

    def get_texture_category(self, imagename):
        """Return texture category from imagename.

        e.g. 'XXX_dog10-bird2.png' -> 'bird
        '"""
        assert type(imagename) is str

        # remove unneccessary words
        a = imagename.split("_")[-1]
        # remove .png etc.
        b = a.split(".")[0]
        # get texture category (last word)
        c = b.split("-")[-1]
        # remove number, e.g. 'bird2' -> 'bird'
        d = ''.join([i for i in c if not i.isdigit()])
        return d