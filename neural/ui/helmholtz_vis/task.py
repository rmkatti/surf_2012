# Standard library imports.
import os.path

# System library imports.
from pyface.api import FileDialog, MessageDialog, OK
from pyface.tasks.action.api import SGroup, SMenu, SMenuBar, \
    EditorAction, TaskAction
from pyface.tasks.api import Task, TaskLayout, PaneItem, VSplitter, \
    AdvancedEditorAreaPane, IEditorAreaPane, TraitsDockPane
from traits.api import Bool, Button, DelegatesTo, Directory, Enum, Instance, \
    Property
from traitsui.api import View, HGroup, VGroup, Item, Label, EnumEditor, \
    ImageEnumEditor, InstanceEditor, TabularEditor, spring
from traitsui.tabular_adapter import TabularAdapter

# Local imports
from neural.runner.api import Runner
from neural.utils.serialize import load
from editors import LayersEditor


class HelmholtzVisTask(Task):

    #### 'Task' interface #####################################################

    id = 'neural.helmholtz_vis'
    name = 'Helmholtz Machine Visualization'

    menu_bar = SMenuBar(
        SMenu(SGroup(TaskAction(name = '&Open',
                                method = 'open',
                                accelerator = 'Ctrl+O'),
                     id='OpenGroup'),
              SGroup(TaskAction(name = '&Save Image',
                                method = 'save',
                                accelerator = 'Ctrl+S',
                                enabled_name = 'editor_area.active_editor'),
                     id='SaveGroup'),
              SGroup(EditorAction(name = '&Close',
                                  method = 'close',
                                  accelerator = 'Ctrl+W'),
                     id='ExitGroup'),
              id='File', name='&File'),
        SMenu(id='View', name='&View'))

    #### 'HelmholtzVisTask' interface #########################################
    
    default_directory = Directory
    editor_area = Instance(IEditorAreaPane)

    ###########################################################################
    # 'Task' interface.
    ###########################################################################

    def create_central_pane(self):
        self.editor_area = editor_area = AdvancedEditorAreaPane()
        editor_area.register_factory(LayersEditor, 
                                     lambda obj: isinstance(obj, Runner))
        return editor_area

    def create_dock_panes(self):
        return [ SamplingPane(), ToolsPane() ]

    ###########################################################################
    # 'HelmholtzVisTask' interface.
    ###########################################################################

    def open(self):
        wildcard = FileDialog.create_wildcard('JSON files', '*.json')
        dialog = FileDialog(action = 'open',
                            default_directory = self.default_directory,
                            parent = self.window.control,
                            wildcard = wildcard)
        if dialog.open() == OK:
            runner = load(dialog.path)
            runner.outfile = dialog.path
            self.editor_area.edit(runner)

    def save(self):
        extensions = [ '*.bmp', '*.gif', '*.jpg', '*.pdf',
                       '*.png', '*.svg', '*.tif', '*.xbm' ]
        wildcard = FileDialog.create_wildcard('From file name', extensions)
        dialog = FileDialog(action = 'save as',
                            default_directory = self.default_directory,
                            parent = self.window.control,
                            wildcard = wildcard)
        if dialog.open() == OK:
            filename = dialog.path
            extension = os.path.splitext(filename)[1]
            if not extension:
                extension = '.png'
                filename += extension
            try:
                # FIXME: Expose size and background color?
                self.editor_area.active_editor.save(filename, bgcolor='white')
            except Exception as exc:
                msg = 'Failed to save image in %s format' % extension.upper()
                dialog = MessageDialog(title = 'Error saving',
                                       message = msg,
                                       detail = str(exc),
                                       parent = self.window.control,
                                       severity = 'error')
                dialog.open()

    ###########################################################################
    # Protected interface.
    ###########################################################################

    #### Trait initializers ###################################################

    def _default_directory_default(self):
        return os.path.expanduser('~')

    def _default_layout_default(self):
        return TaskLayout(
            left=VSplitter(PaneItem('neural.helmholtz_vis.sampling_pane'),
                           PaneItem('neural.helmholtz_vis.tools_pane')))


class SamplingPane(TraitsDockPane):

    id = 'neural.helmholtz_vis.sampling_pane'
    name = 'Sampling'

    machine_model = Enum('generative', 'recognition')
    clamp_top_units = Bool(False)

    active_editor = Property(depends_on='task.editor_area.active_editor')
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
                           Item('sample_button',
                                enabled_when = 'active_editor'),
                           show_labels = False),
                    resizable = True)

    def _get_active_editor(self):
        editor = self.task.editor_area.active_editor
        return editor if isinstance(editor, LayersEditor) else None

    def _sample_button_fired(self):
        if self.active_editor:
            self.active_editor.sample(model = self.machine_model,
                                      clamp_top_units = self.clamp_top_units)


class ToolsPane(TraitsDockPane):
    
    id = 'neural.helmholtz_vis.tools_pane'
    name = 'Tools'

    active_editor = Property(depends_on='task.editor_area.active_editor')

    def default_traits_view(self):
        model_editor = EnumEditor(values = {'generative': 'Generative',
                                            'recognition': 'Recognition'})
        style_editor = EnumEditor(values = {'heat': 'Heat map',
                                            'hinton': 'Hinton diagram'})
        tool_editor = ImageEnumEditor(values = {'activate': 'search',
                                                'toggle': 'pencil'},
                                      cols = 4,
                                      klass = ToolsPane)
        shapes_editor = TabularEditor(adapter = LayerShapesAdapter(),
                                      operations = ['edit'])
        view = View(VGroup(Item('tool',
                                editor = tool_editor,
                                style = 'custom'),
                           HGroup(Label('Inspector model:'),
                                  Item('weights_model',
                                       editor = model_editor,
                                       show_label = False)),
                           HGroup(Label('Inspector style:'),
                                  Item('weights_style',
                                       editor = style_editor,
                                       show_label = False)),
                           Label('Layer shapes:'),
                           Item('layer_shapes',
                                editor = shapes_editor),
                           show_labels = False),
                    resizable = True)
        return View(Item('active_editor',
                         editor = InstanceEditor(view=view),
                         show_label = False,
                         style = 'custom'),
                    resizable = True)

    def _get_active_editor(self):
        editor = self.task.editor_area.active_editor
        return editor if isinstance(editor, LayersEditor) else None


class LayerShapesAdapter(TabularAdapter):
    
    columns = ['Rows', 'Columns']
    alignment = 'right'

    def _set_text(self, value):
        machine = self.object.machine
        try:
            value = int(value)
            assert value > 0
        except (AssertionError, ValueError):
            return

        if machine:
            size = machine.topology[self.row]
            if size % value == 0:
                self.item[self.column_id] = value
                self.item[int(not(self.column_id))] = size / value
        else:
            self.item[self.column_id] = value
        # FIXME: Cleaner way to propagate updates.
        self.object._layer_shapes_changed()
