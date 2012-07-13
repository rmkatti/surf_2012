# System library imports
import numpy as np
from traits.api import Any, Bool, Button, Enum, DelegatesTo, HasTraits, \
    Instance, Int, List, Tuple, on_trait_change
from traitsui.api import View, HGroup, VGroup, Item, Label, EnumEditor, spring

# Local imports
from neural.helmholtz import HelmholtzMachine
from units_plot import UnitsPlot


class HelmholtzVis(HasTraits):

    machine = Any # Instance(HelmholtzMachine)
    layer_shapes = List(Tuple(Int, Int))
    pixel_size = DelegatesTo('plot')

    model = Enum('generative', 'recognition')
    clamp_top_units = Bool(False)

    plot = Instance(UnitsPlot)
    sample_button = Button(label='Sample!')

    def default_traits_view(self):
        model_editor = EnumEditor(
            values = {'generative': 'Generative model',
                      'recognition': 'Recognition model'})
        return View(
            HGroup(VGroup(HGroup(Label('Sample from:'),
                                 Item('model',
                                      editor = model_editor,
                                      show_label = False)),
                          HGroup(Label('Clamp top-level units?'),
                                 Item('clamp_top_units',
                                      show_label = False),
                                 visible_when = "model == 'generative'"),
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
        if self.model == 'generative':
            data = None
            if self.clamp_top_units:
                data = self.plot.layers[0].flatten()
            layers = self.machine.sample_generative_dist(all_layers=True, 
                                                         top_units=data)
        elif self.model == 'recognition':
            data = self.plot.layers[-1].flatten()
            layers = self.machine.sample_recognition_dist(data)
        
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
