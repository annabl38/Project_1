import nltk
# You may have to do this if you haven't yet.
# nltk.download('punkt')
# This is one of the libraries/functions that the nltk developers used in their code.
from six import text_type

# This is another custom function they wrote thats needed in the plotting method.
def _get_kwarg(kwargs, key, default):
    if key in kwargs:
        arg = kwargs[key]
        del kwargs[key]
    else:
        arg = default
    return arg

# Here we create a custom class that extends the FreqDist class that nltk provides.
# This essentially makes a copy of the FreqDist class, and then you can modify it with your own code.
class CustomFreqDist(nltk.probability.FreqDist):
	# Here we create our own plotting function.
	# This is where you can modify the chart with whatever you want.
    def custom_plot(self, *args, **kwargs):
        """
        Plot samples from the frequency distribution
        displaying the most frequent sample first.  If an integer
        parameter is supplied, stop after this many samples have been
        plotted.  For a cumulative plot, specify cumulative=True.
        (Requires Matplotlib to be installed.)

        :param title: The title for the graph
        :type title: str
        :param cumulative: A flag to specify whether the plot is cumulative (default = False)
        :type title: bool
        """
        try:
            from matplotlib import pylab
        except ImportError:
            raise ValueError(
                'The plot function requires matplotlib to be installed.'
                'See http://matplotlib.org/'
            )

        if len(args) == 0:
            args = [len(self)]
        samples = [item for item, _ in self.most_common(*args)]

        cumulative = _get_kwarg(kwargs, 'cumulative', False)
        percents = _get_kwarg(kwargs, 'percents', False)
        if cumulative:
            freqs = list(self._cumulative_frequencies(samples))
            ylabel = "Cumulative Counts"
            if percents:
                freqs = [f / freqs[len(freqs) - 1] * 100 for f in freqs]
                ylabel = "Cumulative Percents"
        else:
            freqs = [self[sample] for sample in samples]
            ylabel = "Counts"
        # percents = [f * 100 for f in freqs]  only in ProbDist?

        pylab.grid(True, color="silver")
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 2
        if "title" in kwargs:
            pylab.title(kwargs["title"])
            del kwargs["title"]
        pylab.plot(freqs, **kwargs)
        pylab.xticks(range(len(samples)), [text_type(s) for s in samples], rotation=90)
        pylab.xlabel("Words")
        pylab.ylabel(ylabel)
		# Here I added the title.
        pylab.show()

