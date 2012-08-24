# System library imports.
from sklearn.grid_search import GridSearchCV
from traits.api import Any, Dict, File, Float, Instance, Int, List, Str, Type

# Local imports.
from neural.runner.api import Runner, EstimatorNeuralRunner
from mnist import binarize_mnist_images, read_mnist


class GridSearchRunner(Runner):
    """ Grid search on the hyperparameters of the digits networks.
    """

    # Configuration.
    cls = Type(config=True, config_default_module='neural.api',
               desc="class of the machine, e.g. 'LadderedHelmholtzMachine' " \
                   "or 'my_package.custom_machine.CustomMachine'")
    cls_args = Dict(Str, Any, config=True,
                    desc="optional keyword arguments for the class constructor")
    data_path = File(config=True, transient=True,
                     desc="path to MNIST data files")

    cv = Int(5, config=True, desc="number of folds for cross-validation")
    jobs = Int(1, config=True, desc="number of jobs to run in parallel")
    param_grid = Dict(Str, List, config=True,
                      desc="mapping from hyperparameter names to lists of " \
                          "values to search")

    # Runtime.
    base_runner = Instance(EstimatorNeuralRunner)

    # Results.
    param_scores = List
    best_params = Dict(Str, Any)
    best_score = Float
    
    def run(self):
        imgs, labels = read_mnist(path=self.data_path, training=True)
        imgs = binarize_mnist_images(imgs)

        gs = GridSearchCV(self.base_runner, self.param_grid,
                          n_jobs = self.jobs, pre_dispatch = '2*n_jobs',
                          cv = self.cv, verbose = self.verbose)
        gs.fit(imgs, labels)

        self.param_scores = gs.grid_scores_
        self.best_params = gs.best_params_
        self.best_score = gs.best_score_
                         
    def _base_runner_default(self):
        from supervised import SupervisedDigitsRunner
        return SupervisedDigitsRunner(cls=self.cls, cls_args=self.cls_args)

    def _cls_default(self):
        from neural.api import HelmholtzMachine
        return HelmholtzMachine

    def _param_grid_default(self):
        return dict(
            anneal = [0.0, 0.5, 1.0, 5.0],
            #base_variance = [1.0, 2.0, 4.0, 5.0, 10.0],
            rate = [0.1],
            topology = [ [16, 64, 64, 28**2],
                         [32, 64, 64, 28**2],
                         [16, 32, 32, 28**2],
                         [16, 128, 128, 28**2] ],
        )


def main(args = None):
    runner = GridSearchRunner()
    runner.main(args=args)
    print 'Best parameters = {}'.format(runner.best_params)
    print 'Best error rate = {:%}'.format(1.0 - runner.best_score)

if __name__ == '__main__':
    main()
