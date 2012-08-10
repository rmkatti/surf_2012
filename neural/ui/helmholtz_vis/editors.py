# Standard library imports.
import os.path

# System library imports.
import numpy as np
from pyface.tasks.api import TraitsEditor
from traits.api import Any, Array, Bool, DelegatesTo, Instance
from traitsui.api import View, Item, InstanceEditor

# Local imports
from neural.helmholtz import HelmholtzMachine
from neural.runner.api import Runner
from neural.ui.api import UnitsPlot


class LayersEditor(TraitsEditor):

    #### 'Editor' interface ###################################################

    obj = Instance(Runner)

    #### 'LayersEditor' interface #############################################

    machine = Any # Instance(HelmholtzMachine)

    layer_shapes = Array(dtype=int, shape=(None,2))
    tool = DelegatesTo('plot')

    activated = DelegatesTo('plot')
    plot = Instance(UnitsPlot)
    traits_view = View(Item('plot',
                            show_label = False,
                            style = 'custom'),
                       resizable = True)

    ###########################################################################
    # 'LayersEditor' interface.
    ###########################################################################

    def edit_weights(self, layer, row, col):
        W = self.machine.G[layer]
        w = W[row * self.layer_shapes[layer][0] + col]
        w = w.reshape(self.layer_shapes[layer+1])

        name = 'Layer {}, Unit ({},{})'.format(layer+1, row+1, col+1)
        title = '{} - {}'.format(self.name, name)
        editor = WeightsEditor(obj=w, name=name, title=title)
        self.editor_area.add_editor(editor)
        return editor

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
        return UnitsPlot(tool = 'toggle')

    #### Trait change handlers ################################################

    def _activated_fired(self, idx):
        self.edit_weights(*idx)

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


class WeightsEditor(TraitsEditor):

    #### 'Editor' interface ###################################################

    obj = Array(dtype=np.double, shape=(None,None))

    #### 'LayersEditor' interface #############################################

    plot = Any
    title = DelegatesTo('plot')
    traits_view = View(Item('plot',
                            editor = InstanceEditor(),
                            show_label = False,
                            style = 'custom'),
                       resizable = True)
    
    ###########################################################################
    # Protected interface.
    ###########################################################################

    #### Trait initializers ###################################################

    def _plot_default(self):
        from neural.ui.api import HintonPlot, WeightsPlot
        return HintonPlot()    
        #return WeightsPlot()

    #### Trait change handlers ################################################

    def _obj_changed(self):
        self.plot.weights = self.obj
