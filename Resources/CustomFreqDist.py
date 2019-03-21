import nltk
from six import text_type

def _get_kwarg(kwargs, key, default):
    if key in kwargs:
        arg = kwargs[key]
        del kwargs[key]
    else:
        arg = default
    return arg

class CustomFreqDist(nltk.probability.FreqDist):
    def custom_plot(self, *args, **kwargs):
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
        return pylab

