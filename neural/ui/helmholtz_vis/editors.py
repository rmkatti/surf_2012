# Standard library imports.
import os.path

# System library imports.
import numpy as np
from pyface.tasks.api import TraitsEditor
from traits.api import Any, Array, Bool, Instance
from traitsui.api import View, Item

# Local imports
from neural.helmholtz import HelmholtzMachine
from neural.runner.api import Runner
from neural.ui.units_plot import UnitsPlot


class LayersEditor(TraitsEditor):

    #### 'Editor' interface ###################################################

    obj = Instance(Runner)

    #### 'LayersEditor' interface #############################################

    machine = Any # Instance(HelmholtzMachine)

    layer_shapes = Array(dtype=int, shape=(None,2))
    plot = Instance(UnitsPlot)

    traits_view = View(Item('plot',
                            show_label = False,
                            style = 'custom'),
                       resizable = True)

    ###########################################################################
    # 'LayersEditor' interface.
    ###########################################################################

    def sample(self, model = 'generative', clamp_top_units = False):
        plot = self.plot
        if model == 'generative':
            data = None
            if clamp_top_units:
                data = plot.layers[0].flatten()
            layers = self.machine.sample_generative_dist(all_layers=True, 
                                                         top_units=data)
        elif model == 'recognition':
            data = plot.layers[-1].flatten()
            layers = self.machine.sample_recognition_dist(data)
        else:
            raise ValueError('Unknown model type %r' % model)
        plot.layers = map(np.reshape, layers, self.layer_shapes)

    ###########################################################################
    # Protected interface.
    ###########################################################################

    #### Trait initializers ###################################################

    def _plot_default(self):
        return UnitsPlot(editable = True)

    #### Trait change handlers ################################################

    def _layer_shapes_changed(self):
        plot = self.plot
        if plot.layers:
            plot.layers = map(np.reshape, plot.layers, self.layer_shapes)

    def _machine_changed(self):
        machine = self.machine
        plot = self.plot
        plot.layers = []
        if machine:
            layer_topology = np.prod(self.layer_shapes, axis=1)
            if not np.all(layer_topology == machine.topology):
                self.layer_shapes = [ (1,size) for size in machine.topology ]
            plot.layers = map(np.zeros, self.layer_shapes)

    def _obj_changed(self):
        self.machine = getattr(self.obj, 'machine', None)
        self.name = os.path.basename(self.obj.outfile)
