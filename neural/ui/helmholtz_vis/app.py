from __future__ import print_function

# Standard library imports.
import logging
import sys

# System library imports.
from traits.etsconfig.api import ETSConfig
from pyface.tasks.api import TaskWindowLayout
from envisage.ui.tasks.api import TasksApplication

# Plugin imports.
from envisage.core_plugin import CorePlugin
from envisage.ui.tasks.tasks_plugin import TasksPlugin
from plugin import HelmholtzVisPlugin

# XXX: Hide warnings from Qt about setting negative sizes. These are harmless
# and I don't have time to track them down.
if ETSConfig.toolkit == 'qt4':

    def warning_filter(msg_type, msg):
        if msg_type == QtCore.QtWarningMsg and 'negative size' in msg.lower():
            return
        print(msg, file=sys.stderr)

    from pyface.qt import QtCore
    QtCore.qInstallMsgHandler(warning_filter)


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
    main(sys.argv)
