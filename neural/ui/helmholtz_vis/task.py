# System library imports.
import numpy as np
from pyface.api import FileDialog, OK
from pyface.tasks.action.api import SGroup, SMenu, SMenuBar, TaskAction
from pyface.tasks.api import Task, TaskLayout, PaneItem, VSplitter, \
    TraitsTaskPane, TraitsDockPane
from traits.api import Any, Array, Bool, Button, DelegatesTo, Enum, Instance
from traitsui.api import View, HGroup, VGroup, Item, Label, EnumEditor, \
    RangeEditor, TabularEditor, spring
from traitsui.tabular_adapter import TabularAdapter

# Local imports
from neural.helmholtz import HelmholtzMachine
from neural.ui.units_plot import UnitsPlot
from neural.utils.serialize import load


class HelmholtzVisTask(Task):

    #### 'Task' interface #####################################################

    id = 'neural.helmholtz_vis'
    name = 'Helmholtz Machine Visualization'

    menu_bar = SMenuBar(
        SMenu(TaskAction(name = 'Open File',
                         method = 'open',
                         accelerator = 'Ctrl+O'),
              id='File', name='&File'))

    #### 'HelmholtzVisTask' interface #########################################

    machine = Any # Instance(HelmholtzMachine)

    layer_shapes = Array(dtype=int, shape=(None,2))
    plot = Instance(UnitsPlot)

    ###########################################################################
    # 'Task' interface.
    ###########################################################################

    def create_central_pane(self):
        return CentralPane()

    def create_dock_panes(self):
        return [ SamplingPane(), LayerDisplayPane() ]

    ###########################################################################
    # 'HelmholtzVisTask' interface.
    ###########################################################################

    def open(self):
        dialog = FileDialog(action = 'open',
                            parent = self.window.control,
                            wildcard = 'JSON files (*.json)|*.json|')
        if dialog.open() == OK:
            runner = load(dialog.path)
            self.machine = getattr(runner, 'machine', None)

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

    def _default_layout_default(self):
        return TaskLayout(
            left=VSplitter(PaneItem('neural.helmholtz_vis.sampling_pane'),
                           PaneItem('neural.helmholtz_vis.layer_display_pane')))

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


class CentralPane(TraitsTaskPane):

    id = 'neural.helmholtz_vis.central_pane'
    name = 'Helmholtz Visualization Pane'

    traits_view = View(Item('object.task.plot',
                            show_label = False,
                            style = 'custom'),
                       resizable = True)


class SamplingPane(TraitsDockPane):

    id = 'neural.helmholtz_vis.sampling_pane'
    name = 'Sampling'

    machine_model = Enum('generative', 'recognition')
    clamp_top_units = Bool(False)

    sample_button = Button(label='Sample!')

    def default_traits_view(self):
        model_editor = EnumEditor(values = {'generative':'Generative model',
                                            'recognition':'Recognition model'})
        return View(VGroup(HGroup(Label('Sample from:'),
                                  Item('machine_model',
                                       editor = model_editor,
                                       show_label = False)),
                           HGroup(Label('Clamp top-level units?'),
                                  Item('clamp_top_units',
                                       show_label = False),
                                  enabled_when = "machine_model=='generative'"),
                           #spring,
                           Item('sample_button'),
                           show_labels = False),
                    resizable = True)

    def _sample_button_fired(self):
        self.task.sample(model = self.machine_model,
                         clamp_top_units = self.clamp_top_units)


class LayerDisplayPane(TraitsDockPane):
    
    id = 'neural.helmholtz_vis.layer_display_pane'
    name = 'Layer Display'

    def default_traits_view(self):
        shapes_editor = TabularEditor(adapter = LayerShapesAdapter(),
                                      operations = ['edit'])
        pixel_size_editor = RangeEditor(is_float = False,
                                        low = 1, high=128,
                                        mode = 'spinner')
        return View(VGroup(HGroup(Label('Pixel size:'),
                                  Item('object.task.plot.pixel_size',
                                       editor = pixel_size_editor,
                                       show_label = False)),
                           Label('Layer shapes:'),
                           Item('object.task.layer_shapes',
                                editor = shapes_editor,
                                show_label = False)),
                    resizable = True)                           


class LayerShapesAdapter(TabularAdapter):
    
    columns = ['Rows', 'Columns']
    alignment = 'right'

    def _set_text(self, value):
        value = int(value)
        machine = self.object.machine
        if machine:
            size = machine.topology[self.row]
            if size % value == 0:
                self.item[self.column_id] = value
                self.item[int(not(self.column_id))] = size / value
        else:
            self.item[self.column_id] = value
        # FIXME: Cleaner way to propagate updates.
        self.object._layer_shapes_changed()
