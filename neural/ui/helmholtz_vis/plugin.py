# System library imports.
from envisage.api import Plugin
from envisage.ui.tasks.api import TaskFactory
from traits.api import List


class HelmholtzVisPlugin(Plugin):

    # Extension point IDs.
    TASKS = 'envisage.ui.tasks.tasks'

    #### 'IPlugin' interface ##################################################

    # The plugin's unique identifier.
    id = 'neural.helmholtz_vis'

    # The plugin's name (suitable for displaying to the user).
    name = 'Helmholtz Machine Visualization'

    #### Contributions to extension points made by this plugin ################

    tasks = List(contributes_to=TASKS)

    ###########################################################################
    # Protected interface.
    ###########################################################################

    def _tasks_default(self):
        from task import HelmholtzVisTask
        return [ TaskFactory(id = 'neural.helmholtz_vis',
                             name = 'Helmholtz Machine Visualization',
                             factory = HelmholtzVisTask), ]
