# Standard library imports.
import logging

# System library imports.
from envisage.ui.tasks.api import TasksApplication
from pyface.tasks.api import TaskWindowLayout

# Plugin imports.
from envisage.core_plugin import CorePlugin
from envisage.ui.tasks.tasks_plugin import TasksPlugin
from plugin import HelmholtzVisPlugin


class HelmholtzVisApplication(TasksApplication):

    #### 'IApplication' interface #############################################

    # The application's globally unique identifier.
    id = 'neural.helmholtz_vis'

    # The application's user-visible name.
    name = 'Helmholtz Machine'

    #### 'TasksApplication' interface #########################################

    # The default window-level layout for the application.
    default_layout = [ TaskWindowLayout('neural.helmholtz_vis',
                                        size = (800,600)) ]


def main(argv):
    """ Run the application.
    """
    logging.basicConfig(level=logging.WARNING)

    plugins = [ CorePlugin(), TasksPlugin(), HelmholtzVisPlugin() ]
    app = HelmholtzVisApplication(plugins=plugins)
    app.run()

    logging.shutdown()


if __name__ == '__main__':
    import sys
    main(sys.argv)
