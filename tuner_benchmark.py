import keras_tuner
import numpy as np
import optuna
import dill
from pathlib import Path

import bayes_opt

from multiprocessing import Pool


def run_objective(tuner, objective, results_dict, **hparams):
    value = objective(**hparams)
    if results_dict.get(tuner) is None:
        results_dict[tuner] = []
    results_dict[tuner].append({**{f'params_{k}': v for k,v in hparams.items()}, 'value': value, 'number': len(results_dict[tuner])})
    return value


def shifted_sine_objective(degree, **hparams):
    r"""
    \sin(\vec{x}) \approx f(\vec{x}, \vec{b}, RH) = \left[
  \begin{array}{cccc}
  x_0^0 & x_0^1 & \cdot & x_0^d \\
  x_1^0 & x_1^1 & \cdot & x_1^d \\
  \vdot & \vdot & \ddot & \vdot \\
  x_n^0 & x_n^1 & \cdot & x_n^d \\
  \end{array}
\right]

\begin{bmatrix}
           b_0 - 10 \\
           b_{1} - 10\\
           \vdots \\
           b_{d} - 10
         \end{bmatrix}
    """
    xs = np.random.uniform(low=0, high=2*np.pi, size=100)
    ys = np.sin(xs)
    degree = int(round(degree))

    transformed_xs = np.mat([xs**n for n in range(degree+1)])
    coefficients = np.mat([hparams[f'b_{n}'] - 10 for n in range(degree+1)])

    model_output = np.array(transformed_xs.transpose()*coefficients.transpose()).flatten()

    error = np.abs(ys-model_output).sum()

    return error


def ackley_function(red_herring=None, **hparams):
    a = 20
    b = 0.2
    c = 2*np.pi
    xs = np.fromiter(hparams.values(), dtype=float) - 10

    n = -b*np.sqrt((xs**2).mean())
    m = np.cos(c*xs).mean()
    
    value = np.e + a - a*np.exp(n) - np.exp(m)

    return value


def run_optuna_rng(num_trials, objective, parameters, results_dict):
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(),
    direction='minimize')

    for i in range(num_trials):
        trial = study.ask()
        for param, [low, high] in parameters.items():
            if param == 'degree':
                trial.suggest_int(param, low, high)
            if 'degree' in trial.params and '_' in param and param.split('_')[-1].isnumeric():
                coefficient_ix = int(param.split('_')[-1])
                if coefficient_ix > trial.params['degree']:
                    continue
            if param not in trial.params:
                trial.suggest_float(param, low, high)
        value = run_objective(tuner='optuna-rng', objective=objective, results_dict=results_dict, **trial.params)
        trial.report(value=value, step=0)
        study.tell(trial, value)


def run_optuna_qmc(num_trials, objective, parameters, results_dict):
    study = optuna.create_study(
        sampler=optuna.samplers.QMCSampler(),
    direction='minimize')

    for i in range(num_trials):
        trial = study.ask()
        for param, [low, high] in parameters.items():
            if param == 'degree':
                trial.suggest_int(param, low, high)
            if 'degree' in trial.params and '_' in param and param.split('_')[-1].isnumeric():
                coefficient_ix = int(param.split('_')[-1])
                if coefficient_ix > trial.params['degree']:
                    continue
            if param not in trial.params:
                trial.suggest_float(param, low, high)
        value = run_objective(tuner='optuna-qmc', objective=objective, results_dict=results_dict, **trial.params)
        trial.report(value=value, step=0)
        study.tell(trial, value)


def run_optuna_rngtpe(num_trials, objective, parameters, results_dict):
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(),
    direction='minimize')

    for i in range(num_trials):
        if i == max(10, 3*len(parameters)) \
            and type(study.sampler) == optuna.samplers.RandomSampler:
            study.sampler = optuna.samplers.TPESampler()
        trial = study.ask()
        for param, [low, high] in parameters.items():
            if param == 'degree':
                trial.suggest_int(param, low, high)
            if 'degree' in trial.params and '_' in param and param.split('_')[-1].isnumeric():
                coefficient_ix = int(param.split('_')[-1])
                if coefficient_ix > trial.params['degree']:
                    continue
            if param not in trial.params:
                trial.suggest_float(param, low, high)
        value = run_objective(tuner='optuna-rngtpe', objective=objective, results_dict=results_dict, **trial.params)
        trial.report(value=value, step=0)
        study.tell(trial, value)


def run_optuna_qmctpe(num_trials, objective, parameters, results_dict):
    study = optuna.create_study(
        sampler=optuna.samplers.QMCSampler(),
    direction='minimize')

    for i in range(num_trials):
        if i == 32 and type(study.sampler) == optuna.samplers.QMCSampler:
            study.sampler = optuna.samplers.TPESampler()
        trial = study.ask()
        for param, [low, high] in parameters.items():
            if param == 'degree':
                trial.suggest_int(param, low, high)
            if 'degree' in trial.params and '_' in param and param.split('_')[-1].isnumeric():
                coefficient_ix = int(param.split('_')[-1])
                if coefficient_ix > trial.params['degree']:
                    continue
            if param not in trial.params:
                trial.suggest_float(param, low, high)
        value = run_objective(tuner='optuna-qmctpe', objective=objective, results_dict=results_dict, **trial.params)
        trial.report(value=value, step=0)
        study.tell(trial, value)


def run_keras_tuner_rng(ix, num_trials, objective, parameters, results_dict):
    class MyTuner(keras_tuner.RandomSearch):
        def run_trial(self, trial, *args, **kwargs):
            # Get the hp from trial.
            hp = trial.hyperparameters
            # Define "x" as a hyperparameter.
            for param, [low, high] in parameters.items():
                if param == 'degree':
                    hp.Int(param, low, high)
                if 'degree' in hp.values and '_' in param and param.split('_')[-1].isnumeric():
                    coefficient_ix = int(param.split('_')[-1])
                    if coefficient_ix > hp.values['degree']:
                        continue
                if param not in hp.values:
                    hp.Float(param, min_value=low, max_value=high)
            # Return the objective value to minimize.
            return run_objective(tuner='keras-tuner-rng', objective=objective, results_dict=results_dict, **hp.values)

    tuner = MyTuner(
        # No hypermodel or objective specified.
        max_trials=num_trials,
        overwrite=True,
        directory=f"my_dir_{ix}",
        project_name="tune_anything",
    )

    # No need to pass anything to search()
    # unless you use them in run_trial().
    tuner.search(verbose=0)


def run_keras_tuner_bayesian(ix, num_trials, objective, parameters, results_dict):
    class MyTuner(keras_tuner.BayesianOptimization):
        def run_trial(self, trial, *args, **kwargs):
            # Get the hp from trial.
            hp = trial.hyperparameters
            # Define "x" as a hyperparameter.
            for param, [low, high] in parameters.items():
                if param == 'degree':
                    hp.Int(param, low, high)
                if 'degree' in hp.values and '_' in param and param.split('_')[-1].isnumeric():
                    coefficient_ix = int(param.split('_')[-1])
                    if coefficient_ix > hp.values['degree']:
                        continue
                if param not in hp.values:
                    hp.Float(param, min_value=low, max_value=high)
            # Return the objective value to minimize.
            return run_objective(tuner='keras-tuner-bayesian', objective=objective, results_dict=results_dict, **hp.values)

    tuner = MyTuner(
        # No hypermodel or objective specified.
        max_trials=num_trials,
        overwrite=True,
        directory=f"my_dir_{ix}",
        project_name="tune_anything",
    )

    # No need to pass anything to search()
    # unless you use them in run_trial().
    tuner.search(verbose=0)


def run_bayesian_optimization(num_trials, objective, parameters, results_dict):
    optimizer = bayes_opt.BayesianOptimization(
        f=lambda **kwargs: -run_objective(tuner='bayesian-optimization', objective=objective, results_dict=results_dict, **kwargs),
        pbounds=parameters,
        verbose=1,
    )

    optimizer.maximize(
        n_iter=num_trials
    )


studies = {
    'x_10pi': dict(
        num_studies=50,
        num_trials=100,
        parameters = {
            'x': [-100,100],
        },
        objective=lambda x, **kwargs: (x-10*np.pi)**2,
    ),

    'x_10pi_y_5pi': dict(
        num_studies=50,
        num_trials=100,
        parameters={
            'x': [-100, 100],
            'y': [-100, 100],
        },
        objective=lambda x, y, **kwargs: (x-10*np.pi)**2 + abs(y-5*np.pi)
    ),

    'x_10pi_y_5pi_r_h': dict(
        num_studies=50,
        num_trials=100,
        parameters={
            'x': [-100, 100],
            'y': [-100, 100],
            'red_herring': [0, 1],
        },
        objective=lambda x, y, **kwargs: (x-10*np.pi)**2 + abs(y-5*np.pi)
    ),

    '2x_10pi_y_5pi_r_h': dict(
        num_studies=50,
        num_trials=100,
        parameters={
            'x_0': [-100, 100],
            'x_1': [-100, 100],
            'y': [-100, 100],
            'red_herring': [0, 1],
        },
        objective=lambda x_0, x_1, y, **kwargs: (x_0-10*np.pi)**2 + (x_1-10*np.pi)**2 + abs(y-5*np.pi)
    ),

    'x_10pi_2y_5pi_r_h': dict(
        num_studies=50,
        num_trials=100,
        parameters={
            'x': [-100, 100],
            'y_0': [-100, 100],
            'y_1': [-100, 100],
            'red_herring': [0, 1],
        },
        objective=lambda x, y_0, y_1, **kwargs: (x-10*np.pi)**2 + abs(y_0-5*np.pi) + abs(y_1-5*np.pi)
    ),

    'v_20pi_y_5pi_r_h': dict(
        num_studies=50,
        num_trials=100,
        parameters={
            'v': [-100, 100],
            'y': [-100, 100],
            'red_herring': [0, 1],
        },
        objective=lambda v, y, **kwargs: (v-20*np.pi)**8 + (y-5*np.pi)
    ),

    'x_10pi_y_5pi_z_2pi_r_h': dict(
        num_studies=50,
        num_trials=100,
        parameters={
            'x': [-100, 100],
            'y': [-100, 100],
            'z': [-100, 100],
            'red_herring': [0, 1],
        },
        objective=lambda x, y, z, **kwargs: (x-10*np.pi)**4 + (y-5*np.pi)**2 + abs(z-2*np.pi)
    ),

    'shifted_sine_approx': dict(
        num_studies=100,
        num_trials=100,
        parameters={
            'red_herring': [0, 1],
            'degree': [3,5],
            'b_0': [-100, 100],
            'b_1': [-100, 100],
            'b_2': [-100, 100],
            'b_3': [-100, 100],
            'b_4': [-100, 100],
            'b_5': [-100, 100],
        },
        objective=shifted_sine_objective
    ),

    'v_20pi_w_15pi_x_10pi_y_5pi_z_2pi_r_h': dict(
        num_studies=50,
        num_trials=100,
        parameters={
            'v': [-100, 100],
            'w': [-100, 100],
            'x': [-100, 100],
            'y': [-100, 100],
            'z': [-100, 100],
            'red_herring': [0, 1],
        },
        objective=lambda v, w, x, y, z, **kwargs: (v-20*np.pi)**8 + (w-15*np.pi)**6 + (x-10*np.pi)**4 + (y-5*np.pi)**2 + abs(z-2*np.pi)
    ),
}

def run_trial(i, study_parameters):
    study_parameters = dill.loads(study_parameters)
    print(f'Running study #{i+1}')
    tuning_results = dict()

    run_bayesian_optimization(
        num_trials=study_parameters['num_trials'],
        parameters=study_parameters['parameters'],
        objective=study_parameters['objective'],
        results_dict=tuning_results,
    )

    print('Running Optuna (RNG)')
    run_optuna_rng(
        num_trials=study_parameters['num_trials'],
        parameters=study_parameters['parameters'],
        objective=study_parameters['objective'],
        results_dict=tuning_results,
    )

    print('Running Optuna (QMC)')
    run_optuna_qmc(
        num_trials=study_parameters['num_trials'],
        parameters=study_parameters['parameters'],
        objective=study_parameters['objective'],
        results_dict=tuning_results,
    )

    print('Running Optuna (RNG -> TPE)')
    run_optuna_rngtpe(
        num_trials=study_parameters['num_trials'],
        parameters=study_parameters['parameters'],
        objective=study_parameters['objective'],
        results_dict=tuning_results,
    )

    print('Running Optuna (QMC -> TPE)')
    run_optuna_qmctpe(
        num_trials=study_parameters['num_trials'],
        parameters=study_parameters['parameters'],
        objective=study_parameters['objective'],
        results_dict=tuning_results,
    )

    print('Running Keras-Tuner (RNG)')
    run_keras_tuner_rng(
        ix=i,
        num_trials=study_parameters['num_trials'],
        parameters=study_parameters['parameters'],
        objective=study_parameters['objective'],
        results_dict=tuning_results,
    )

    print('Running Keras-Tuner (bayesian)')
    run_keras_tuner_bayesian(
        ix=i,
        num_trials=study_parameters['num_trials'],
        parameters=study_parameters['parameters'],
        objective=study_parameters['objective'],
        results_dict=tuning_results,
    )

    return tuning_results


if __name__ == '__main__':
    for study_name, study_parameters in studies.items():
        overall_results = []
        print(f'Running study "{study_name}"')

        # Probably not the best way to do this,
        # but it's fast enough that I don't care.
        with Pool(50) as p:
            overall_results = p.starmap(run_trial, [(i, dill.dumps(study_parameters, recurse=True)) for i in range(study_parameters['num_studies'])])

        study_parameters['results'] = overall_results # type: ignore

        with Path(f'tuner_benchmark_results.{study_name}.dill').open('wb') as handle:
            dill.dump(study_parameters, handle)
