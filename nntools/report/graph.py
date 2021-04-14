from collections import OrderedDict
import numpy as np


from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import layout, widgetbox, column
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.models.widgets import RadioButtonGroup


def display_confMat(confMats, labels, text_angle=0, plot_size=800):

    if not isinstance(confMats, dict):
        confMats = {"Confusion Matrix": confMats}

    COLOR =  '#00cc66'
    TOOLS = "hover,save,pan"

    def get_list(cm):
        """
        Simple function to flatten the confusion matrix
        :param cm:
        :return:
        """
        predicted = []
        actual = []
        count = []
        color = []
        alpha = []
        ratios = []
        N = len(labels)
        confMatNormalized = cm / cm.sum(0)
        for coli in range(N):
            for rowi in range(N):
                predicted.append(labels[coli])
                actual.append(labels[rowi])
                count.append(cm[coli, rowi])
                ratio = confMatNormalized[coli, rowi]
                ratios.append(ratio)
                a = min(ratio + 0.01, 1)
                alpha.append(a)
                color.append(COLOR)
        return predicted, actual, count, color, alpha, ratios

    source_data = {}
    for i, (k, v) in enumerate(confMats.items()):
        predicted, actual, count, color, alpha, ratios = get_list(v)
        source_data[str(i) + 'count'] = count
        source_data[str(i) + 'alphas'] = alpha
        source_data[str(i) + 'ratios'] = ratios

        if i == 0:
            source_data['predicted'] = predicted
            source_data['groundtruth'] = actual
            source_data['count'] = ["{:.2e}".format(c) for c in count]
            source_data['colors'] = color
            source_data['alphas'] = alpha
            source_data['ratios'] = ratios

    source = ColumnDataSource(data=source_data)
    p = figure(title='Confusion Matrix',
               x_axis_location="above", tools="hover,save",
               y_range=labels[::-1], x_range=labels)

    p.plot_width = plot_size
    p.plot_height = p.plot_width
    rectwidth = 0.9
    p.rect('predicted', 'groundtruth', rectwidth, rectwidth, source=source,
           color='colors', alpha='alphas', line_width=1)

    p.text(x='predicted', y='groundtruth', text='count', source=source, text_align='center',
           text_baseline='middle')

    p.axis.major_label_text_font_size = "8pt"
    if text_angle:
        p.xaxis.major_label_orientation = np.deg2rad(text_angle)
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.axis_label = 'Predicted'
    p.yaxis.axis_label = 'Groundtruth'

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([
        ('predicted', '@predicted'),
        ('groundtruth', '@groundtruth'),
        ('ratio', '@ratios'),
    ])
    if len(confMats) > 1:
        callback = CustomJS(args=dict(source=source),
                            code=
                            """
                            var data = source.data;
                            var f = cb_obj.active
                            var count = data['count']
                            var alphas = data['alphas']
                            var ratios = data['ratios']

                            for (var i = 0; i < count.length; i++) {
                                count[i] = data[f.toString()+'count'][i]
                                alphas[i] = data[f.toString()+'alphas'][i]
                                ratios[i] = data[f.toString()+'ratios'][i]

                            }
                            source.change.emit();
                            """
                            )
        radio_button_group = RadioButtonGroup(labels=list(confMats.keys()), active=0)
        radio_button_group.js_on_change('active', callback)
        p = column(radio_button_group, p)
    return p





