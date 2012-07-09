# System library imports
import numpy as np
from traits.api import Button, Enum, DelegatesTo, HasTraits, Instance, Int, \
    List, Tuple, on_trait_change
from traitsui.api import View, HGroup, VGroup, Item, Label, EnumEditor, spring

# Local imports
from neural.helmholtz import HelmholtzMachine, sample_generative_dist, \
    sample_recognition_dist
from units_plot import UnitsPlot


class HelmholtzVis(HasTraits):

    machine = Instance(HelmholtzMachine)
    layer_shapes = List(Tuple(Int, Int))
    pixel_size = DelegatesTo('plot')

    model = Enum('generative', 'recognition')

    plot = Instance(UnitsPlot)
    sample_button = Button(label='Sample!')

    def trait_view(self, name=None):
        model_editor = EnumEditor(
            values = {'generative': 'Generative model',
                      'recognition': 'Recognition model'})
        return View(
            HGroup(VGroup(HGroup(Label('Sample from:'),
                                 Item('model',
                                      editor = model_editor,
                                      show_label = False)),
                          spring,
                          Item('sample_button'),
                          show_labels = False),
                   Item('plot',
                        show_label = False,
                        style = 'custom',
                        width = 0.75),
                   layout = 'split'),
            resizable = True,
            title = 'Helmholtz Machine')

    def _plot_default(self):
        return UnitsPlot(editable = True)

    @on_trait_change('sample_button')
    def sample(self):
        machine = self.machine
        if self.model == 'generative':
            layers = sample_generative_dist(machine.G, machine.G_bias, 1, 
                                            all_layers=True)
        elif self.model == 'recognition':
            data = self.plot.layers[-1].flatten()
            hidden = sample_recognition_dist(machine.R, data, 1)
            layers = hidden + [ data ]
        for layer, shape in zip(layers, self.layer_shapes):
            layer.shape = shape
        self.plot.layers = layers

    @on_trait_change('machine, layer_shapes')
    def _reset_plot(self):
        if self.layer_shapes:
            self.plot.layers = [ np.zeros(shape) 
                                 for shape in self.layer_shapes ]
        else:
            self.plot.layers = []
