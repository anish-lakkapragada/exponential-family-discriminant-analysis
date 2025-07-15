from models import Class_Conditional_Density, Normal_LDA_Density, Speeds, EvalResult, EvaluationConfig, ExperimentEvaluation
from helpers import generate_data, METHOD_TO_DENSITY, compute_ece
from custom import run_log_reg_wishart, run_log_reg
from sklearn.metrics import roc_auc_score
from typing import List 
import numpy as np
import time
from rich.panel import Panel
from rich.table import Table 
from rich.console import Console
from rich import print 
from pathlib import Path 

class Evaluator: 
    distributions: List[Class_Conditional_Density]
    num_classes: int 
    class_weight_mles: List[float]

    def __init__(self, distributions: List[Class_Conditional_Density], num_classes: int) -> None: 
        self.distributions = distributions 
        self.num_classes = num_classes

    def fit_class_densities(self, X: np.ndarray, y: np.ndarray) -> Speeds:
         
        distribution_types_lda = [type(distribution) == Normal_LDA_Density for distribution in self.distributions]
        
        using_lda = False 
        if not all(distribution_types_lda) and any(distribution_types_lda): 
            return "Cannot do LDA for a single class."
        elif all(distribution_types_lda): 
            using_lda = True 

        assert len(self.distributions) == self.num_classes, "Not required amount of distributions provided."
        
        speeds = [0] * self.num_classes
        self.class_weight_mles = [0] * self.num_classes

        for class_k in range(self.num_classes): 
            idx = y == class_k 
            self.class_weight_mles[class_k] = np.sum(idx) / X.shape[0]

            # fit the actual density. 
            start_time = time.time()
            self.distributions[class_k].fit_density(X[idx])
            end_time = time.time() 
            speeds[class_k] = end_time - start_time 
        
        assert np.abs(np.sum(self.class_weight_mles) - 1.00) < 1e-5, np.abs(np.sum(self.class_weight_mles) - 1.00)
        
        if using_lda: 
            X2 = X if X.ndim > 1 else X[:, np.newaxis]

            d = X2.shape[1]

            total_scatter = np.zeros((d, d))
            for k, dist in enumerate(self.distributions):
                Xk = X2[y == k]
                mu_k = np.atleast_1d(dist.mean)
                diff = Xk - mu_k
                total_scatter += diff.T @ diff

            N = X2.shape[0]
            pooled_cov = total_scatter / N

            for dist in self.distributions:
                dist.cov_matrix = pooled_cov
        
        return speeds 


    def evaluate(self, X: np.ndarray, y: np.ndarray, plot_auc: bool =  False) -> EvalResult: 
        denom = np.zeros((self.num_classes, y.shape[0]))
        for i, distribution in enumerate(self.distributions): 
            denom[i] = self.class_weight_mles[i] * distribution.evaluate_pdf(X)
        y_pred = denom[1] / np.sum(denom, axis=0)

        return EvalResult(
            # y_pred=y_pred, 
            auc_roc=roc_auc_score(y, y_pred), 
            acc=np.sum(np.round(y_pred) == y) / y.shape[0], 
            ece=compute_ece(y, y_pred, n_bins=10)
        )


if __name__ == "__main__": 
    json_text = Path("config.json").read_text(encoding="utf-8")
    evaluation_cfg = EvaluationConfig.model_validate_json(json_text)
    
    for exp_num, experiment in enumerate(evaluation_cfg.experiments):
        evaluation_table = Table(title=experiment.experiment_name)
        for col in ["Methods", "True Params", "Class Imbalance", "Train AUC", "Train Acc", "Train ECE", "Test AUC", "Test Acc", "Test ECE"]: 
            evaluation_table.add_column(col)

        test_to_tracker_train = {i: ExperimentEvaluation() for i in range(len(experiment.tests))}
        test_to_tracker_test = {i: ExperimentEvaluation() for i in range(len(experiment.tests))}

        for trial in range(experiment.n_trials): 
            X_train, y_train, X_test, y_test = generate_data(experiment.data)

            for t, test in enumerate(experiment.tests): 
                if test.custom: 
                    if test.densities[0] == "log-reg": 
                        eval_result_train, eval_result_test = run_log_reg(X_train, y_train, X_test, y_test)
                        test_to_tracker_train[t].add_result(evaluator.evaluate(X_train, y_train))
                        test_to_tracker_test[t].add_result(evaluator.evaluate(X_test, y_test))
                        break  

                    elif test.densities[0] == "log-reg-matrix":
                        # use logistic regression 
                        eval_result_train, eval_result_test = run_log_reg_wishart(X_train, y_train, X_test, y_test)
                        test_to_tracker_train[t].add_result(evaluator.evaluate(X_train, y_train))
                        test_to_tracker_test[t].add_result(evaluator.evaluate(X_test, y_test))
                        break  

                run_densities = [] 
                for i, method in enumerate(test.densities): 
                    density = METHOD_TO_DENSITY[method]()
                    if test.param_defaults: 
                        param_defaults = test.param_defaults[i]

                        density.set_param_defaults(param_defaults)
                    run_densities.append(density)

                evaluator = Evaluator(num_classes=experiment.data.num_classes, distributions=run_densities)
                speeds = evaluator.fit_class_densities(X_train, y_train)
                test_to_tracker_train[t].add_result(evaluator.evaluate(X_train, y_train))
                test_to_tracker_test[t].add_result(evaluator.evaluate(X_test, y_test))
                
        for t, test in enumerate(experiment.tests): 
            test_tracker_train, test_tracker_test = test_to_tracker_train[t], test_to_tracker_test[t]
            evaluation_table.add_row(",".join(test.densities), 
                                    str(experiment.data.true_params),
                                    ",".join([str(i) for i in experiment.data.class_imbalance_true])
                                    , str(test_tracker_train.get_auc()), str(test_tracker_train.get_acc()), str(test_tracker_train.get_ece())
                                    , str(test_tracker_test.get_auc()), str(test_tracker_test.get_acc()), str(test_tracker_test.get_ece()))
            
        console = Console()
        console.print(evaluation_table)

        if evaluation_cfg.store_dir: 
            store_dir = Path(evaluation_cfg.store_dir)
            store_dir.mkdir(parents=True, exist_ok=True)

            store_file = store_dir / evaluation_cfg.name

            mode = "w" if exp_num == 0 else "a"

            with store_file.open(mode, encoding="utf-8") as f:
                file_console = Console(file=f, width=console.width)
                file_console.print(evaluation_table)

            if exp_num == len(evaluation_cfg.experiments) - 1: 
                config_json = evaluation_cfg.model_dump_json(indent=2)
                with store_file.open("a", encoding="utf-8") as f:
                    f.write(config_json)
                    f.write("\n")