var documenterSearchIndex = {"docs":
[{"location":"python_api/#Python-API","page":"Python API","title":"Python API","text":"","category":"section"},{"location":"python_api/","page":"Python API","title":"Python API","text":"You can directly interface with the Python API for all RAPIDS packages. By default, the following packages are exported:","category":"page"},{"location":"python_api/","page":"Python API","title":"Python API","text":"cupy\ncudf\ncuml\ncugraph\ncusignal\ncuspatial\ncuxfilter\ndask\ndask_cuda\ndask_cudf\nnumpy\npickle","category":"page"},{"location":"python_api/#CUML-Example-Classification","page":"Python API","title":"CUML Example - Classification","text":"","category":"section"},{"location":"python_api/","page":"Python API","title":"Python API","text":"using RAPIDS\nusing PythonCall\n\nX_numpy = numpy.random.rand(100, 5)\ny_numpy = numpy.random.randint(0, 2, 100)\n\nmodel = cuml.LogisticRegression()\nmodel.fit(X_numpy, y_numpy)\npreds_numpy = model.predict(X_numpy)\npreds = pyconvert(Array, preds_numpy)","category":"page"},{"location":"python_api/#CUML-Example-Regression","page":"Python API","title":"CUML Example - Regression","text":"","category":"section"},{"location":"python_api/","page":"Python API","title":"Python API","text":"using RAPIDS\nusing PythonCall\n\nX_numpy = numpy.random.rand(100, 5)\ny_numpy = numpy.random.rand(100)\n\nmodel = cuml.LinearRegression()\nmodel.fit(X_numpy, y_numpy)\npreds_numpy = model.predict(X_numpy)\npreds = pyconvert(Array, preds_numpy)","category":"page"},{"location":"#RAPIDS.jl-Docs","page":"Home","title":"RAPIDS.jl Docs","text":"","category":"section"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"From source:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> ]add https://github.com/tylerjthomas9/RAPIDS.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using Pkg; Pkg.add(url=\"https://github.com/tylerjthomas9/RAPIDS.jl\")","category":"page"},{"location":"cuml/#MLJ-Example-Classification","page":"cuMl","title":"MLJ Example - Classification","text":"","category":"section"},{"location":"cuml/","page":"cuMl","title":"cuMl","text":"using RAPIDS\nusing MLJBase\n\nX = rand(100, 5)\ny = [repeat([0], 50)..., repeat([1], 50)...]\n\nmodel = LogisticRegression()\nmach = machine(model, X, y)\nfit!(mach)\npreds = predict(mach, X)","category":"page"},{"location":"cuml/#MLJ-Example-Regression","page":"cuMl","title":"MLJ Example - Regression","text":"","category":"section"},{"location":"cuml/","page":"cuMl","title":"cuMl","text":"using RAPIDS\nusing MLJBase\n\nX = rand(100, 5)\ny = rand(100)\n\nmodel = LinearRegression()\nmach = machine(model, X, y)\nfit!(mach)\npreds = predict(mach, X)","category":"page"},{"location":"cuml/#MLJ-API","page":"cuMl","title":"MLJ API","text":"","category":"section"},{"location":"cuml/","page":"cuMl","title":"cuMl","text":"CurrentModule = RAPIDS","category":"page"},{"location":"cuml/","page":"cuMl","title":"cuMl","text":"","category":"page"},{"location":"cuml/#Clustering","page":"cuMl","title":"Clustering","text":"","category":"section"},{"location":"cuml/","page":"cuMl","title":"cuMl","text":"    - `KMeans`\n    - `DBSCAN`\n    - `AgglomerativeClustering`\n    - `HDBSCAN`","category":"page"},{"location":"cuml/#Classification","page":"cuMl","title":"Classification","text":"","category":"section"},{"location":"cuml/","page":"cuMl","title":"cuMl","text":"LogisticRegression\nMBSGDClassifier\nRandomForestClassifier\nSVC\nLinearSVC\nKNeighborsClassifier","category":"page"},{"location":"cuml/#RAPIDS.LogisticRegression","page":"cuMl","title":"RAPIDS.LogisticRegression","text":"LogisticRegression\n\nA model type for constructing a logistic regression, based on cuML Classification Methods.jl, and implementing the MLJ model interface.\n\nFrom MLJ, the type can be imported using\n\nLogisticRegression = @load LogisticRegression pkg=cuML Classification Methods\n\nDo model = LogisticRegression() to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in LogisticRegression(penalty=...).\n\nLogisticRegression is a wrapper for the RAPIDS Logistic Regression.\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance model to data with     mach = machine(model, X, y)\n\nwhere\n\nX: any table or array of input features (eg, a DataFrame) whose columns   each have one of the following element scitypes: Continuous\ny: is the target, which can be any AbstractVector whose element   scitype is <:OrderedFactor or <:Multiclass; check the scitype   with scitype(y)\n\nTrain the machine using fit!(mach, rows=...).\n\nHyper-parameters\n\npenalty=\"l2\": Normalization/penalty function (\"none\", \"l1\", \"l2\", \"elasticnet\").\nnone: the L-BFGS solver will be used\nl1: The L1 penalty is best when there are only a few useful features (sparse), and you       want to zero out non-important features. The L-BFGS solver will be used.\nl2: The L2 penalty is best when you have a lot of important features, especially if they       are correlated.The L-BFGS solver will be used.\nelasticnet: A combination of the L1 and L2 penalties. The OWL-QN solver will be used if               l1_ratio>0, otherwise the L-BFGS solver will be used.\n`tol=1e-4': Tolerance for stopping criteria. \nC=1.0: Inverse of regularization strength.\nfit_intercept=true: If True, the model tries to correct for the global mean of y.                        If False, the model expects that you have centered the data.\nclass_weight=\"balanced\": Dictionary or \"balanced\".\nmax_iter=1000: Maximum number of iterations taken for the solvers to converge.\nlinesearch_max_iter=50: Max number of linesearch iterations per outer iteration used in                            the lbfgs and owl QN solvers.\nsolver=\"qn\": Algorithm to use in the optimization problem. Currently only qn is                supported, which automatically selects either L-BFGSor OWL-QN\nl1_ratio=nothing: The Elastic-Net mixing parameter. \nverbose=false: Sets logging level.\n\nOperations\n\npredict(mach, Xnew): return predictions of the target given   features Xnew having the same scitype as X above. Predictions   are class assignments. \npredict_proba(mach, Xnew): return predictions of the target given   features Xnew having the same scitype as X above. Predictions   are probabilistic, but uncalibrated.\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\nmodel: the trained model object created by the RAPIDS.jl package\n\nReport\n\nThe fields of report(mach) are:\n\nclasses_seen: list of target classes actually observed in training\nfeatures: the names of the features encountered in training.\n\nExamples\n\nusing RAPIDS\nusing MLJBase\n\nX = rand(100, 5)\ny = [repeat([0], 50)..., repeat([1], 50)...]\n\nmodel = LogisticRegression()\nmach = machine(model, X, y)\nfit!(mach)\npreds = predict(mach, X)\n\n\n\n\n\n","category":"type"},{"location":"cuml/#RAPIDS.MBSGDClassifier","page":"cuMl","title":"RAPIDS.MBSGDClassifier","text":"MBSGDClassifier\n\nA model type for constructing a mbsgd classifier, based on cuML Classification Methods.jl, and implementing the MLJ model interface.\n\nFrom MLJ, the type can be imported using\n\nMBSGDClassifier = @load MBSGDClassifier pkg=cuML Classification Methods\n\nDo model = MBSGDClassifier() to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in MBSGDClassifier(loss=...).\n\nMBSGDClassifier is a wrapper for the RAPIDS Mini Batch SGD Classifier.\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance model to data with     mach = machine(model, X, y)\n\nwhere\n\nX: any table or array of input features (eg, a DataFrame) whose columns   each have one of the following element scitypes: Continuous\ny: is an AbstractVector finite target.\n\nTrain the machine using fit!(mach, rows=...).\n\nHyper-parameters\n\nloss=\"squared_loss\": Loss function (\"hinge\", \"log\", \"squared_loss\").\nhinge: Linear SVM\nlog: Logistic regression\nsquared_loss: Linear regression\npenalty=\"none\": Normalization/penalty function (\"none\", \"l1\", \"l2\", \"elasticnet\").\nnone: the L-BFGS solver will be used\nl1: The L1 penalty is best when there are only a few useful features (sparse), and you       want to zero out non-important features. The L-BFGS solver will be used.\nl2: The L2 penalty is best when you have a lot of important features, especially if they       are correlated.The L-BFGS solver will be used.\nelasticnet: A combination of the L1 and L2 penalties. The OWL-QN solver will be used if               l1_ratio>0, otherwise the L-BFGS solver will be used.\nalpha=1e-4: The constant value which decides the degree of regularization.\nl1_ratio=nothing: The Elastic-Net mixing parameter. \nbatch_size: The number of samples in each batch.\nfit_intercept=true: If True, the model tries to correct for the global mean of y.                        If False, the model expects that you have centered the data.\nepochs=1000: The number of times the model should iterate through the entire dataset during training.\n`tol=1e-3': The training process will stop if currentloss > previousloss - tol.\nshuffle=true: If true, shuffles the training data after each epoch.\neta0=1e-3: The initial learning rate.\npower_t=0.5: The exponent used for calculating the invscaling learning rate.\nlearning_rate=\"constant: Method for modifying the learning rate during training                           (\"adaptive\", \"constant\", \"invscaling\", \"optimal\")\noptimal: not supported\nconstant: constant learning rate\nadaptive: changes the learning rate if the training loss or the validation accuracy does               not improve for niterno_change epochs. The old learning rate is generally divided by 5.\ninvscaling: eta = eta0 / pow(t, power_t)\nn_iter_no_change=5: the number of epochs to train without any imporvement in the model\nverbose=false: Sets logging level.\n\nOperations\n\npredict(mach, Xnew): return predictions of the target given   features Xnew having the same scitype as X above. Predictions   are class assignments. \npredict_proba(mach, Xnew): return predictions of the target given   features Xnew having the same scitype as X above. Predictions   are probabilistic, but uncalibrated.\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\nmodel: the trained model object created by the RAPIDS.jl package\n\nReport\n\nThe fields of report(mach) are:\n\nclasses_seen: list of target classes actually observed in training\nfeatures: the names of the features encountered in training.\n\nExamples\n\nusing RAPIDS\nusing MLJBase\n\nX = rand(100, 5)\ny = [repeat([0], 50)..., repeat([1], 50)...]\n\nmodel = MBSGDClassifier()\nmach = machine(model, X, y)\nfit!(mach)\npreds = predict(mach, X)\n\n\n\n\n\n","category":"type"},{"location":"cuml/#RAPIDS.RandomForestClassifier","page":"cuMl","title":"RAPIDS.RandomForestClassifier","text":"RandomForestClassifier\n\nA model type for constructing a random forest classifier, based on cuML Classification Methods.jl, and implementing the MLJ model interface.\n\nFrom MLJ, the type can be imported using\n\nRandomForestClassifier = @load RandomForestClassifier pkg=cuML Classification Methods\n\nDo model = RandomForestClassifier() to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in RandomForestClassifier(n_estimators=...).\n\nRandomForestClassifier is a wrapper for the RAPIDS RandomForestClassifier.\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance model to data with     mach = machine(model, X, y)\n\nwhere\n\nX: any table or array of input features (eg, a DataFrame) whose columns   each have one of the following element scitypes: Continuous\ny: is an AbstractVector finite target.\n\nTrain the machine using fit!(mach, rows=...).\n\nHyper-parameters\n\nn_estimators=100: The total number of trees in the forest.\nsplit_creation=2: The criterion used to split nodes\n0 or gini for gini impurity\n1 or entropy for information gain (entropy)\nbootstrap=true: If true, each tree in the forest is built using a bootstrap sample with replacement.\nmax_samples=1.0: Ratio of dataset rows used while fitting each tree.\nmax_depth=16: Maximum tree depth.\nmax_leaves=-1: Maximum leaf nodes per tree. Soft constraint. Unlimited, If -1.\nmax_features=\"auto\": Ratio of number of features (columns) to consider per node split.\nIf type Int then max_features is the absolute count of features to be used.\nIf type Float64 then max_features is a fraction.\nIf auto then max_features=n_features = 1.0.\nIf sqrt then max_features=1/sqrt(n_features).\nIf log2 then max_features=log2(n_features)/n_features.\nIf None, then max_features=1.0.\nn_bins=128: Maximum number of bins used by the split algorithm per feature.\nn_streams=4: Number of parallel streams used for forest building\nmin_samples_leaf=1: The minimum number of samples in each leaf node.\nIf type Int, then min_samples_leaf represents the minimum number.\nIf Float64, then min_samples_leaf represents a fraction and ceil(min_samples_leaf * n_rows)   is the minimum number of samples for each leaf node.\nmin_samples_split=2: The minimum number of samples required to split an internal node.\nIf type Int, then min_samples_split represents the minimum number.\nIf Float64, then min_samples_split represents a fraction and ceil(min_samples_leaf * n_rows)   is the minimum number of samples for each leaf node.\nmin_impurity_decrease=0.0: The minimum decrease in impurity required for node to be split.\nmax_batch_size=4096: Maximum number of nodes that can be processed in a given batch.\nrandom_state=nothing: Seed for the random number generator.\nverbose=false: Sets logging level.\n\nOperations\n\npredict(mach, Xnew): return predictions of the target given   features Xnew having the same scitype as X above. Predictions   are class assignments. \n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\nmodel: the trained model object created by the RAPIDS.jl package\n\nReport\n\nThe fields of report(mach) are:\n\nclasses_seen: list of target classes actually observed in training\nfeatures: the names of the features encountered in training.\n\nExamples\n\nusing RAPIDS\nusing MLJBase\n\nX = rand(100, 5)\ny = [repeat([0], 50)..., repeat([1], 50)...]\n\nmodel = RandomForestClassifier()\nmach = machine(model, X, y)\nfit!(mach)\npreds = predict(mach, X)\n\n\n\n\n\n","category":"type"},{"location":"cuml/#RAPIDS.SVC","page":"cuMl","title":"RAPIDS.SVC","text":"SVC\n\nA model type for constructing a svc, based on cuML Classification Methods.jl, and implementing the MLJ model interface.\n\nFrom MLJ, the type can be imported using\n\nSVC = @load SVC pkg=cuML Classification Methods\n\nDo model = SVC() to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in SVC(C=...).\n\nSVC is a wrapper for the RAPIDS SVC.\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance model to data with     mach = machine(model, X, y)\n\nwhere\n\nX: any table or array of input features (eg, a DataFrame) whose columns   each have one of the following element scitypes: Continuous\ny: is the target, which can be any AbstractVector whose element   scitype is <:OrderedFactor or <:Multiclass; check the scitype   with scitype(y)\n\nTrain the machine using fit!(mach, rows=...).\n\nHyper-parameters\n\nC=1.0: The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.\nkernel=\"rbf\": linear, poly, rbf, sigmoid are supported.\ndegree=3: Degree of polynomial kernel function.\ngamma=\"scale\"\nauto: gamma will be set to 1 / n_features\nscale: gamma will be set to 1 / (n_features * var(X))\ncoef0=0.0: Independent term in kernel function, only signifficant for poly and sigmoid.\ntol=0.001: Tolerance for stopping criterion.\ncache_size=1024.0: Size of the cache during training in MiB.\nclass_weight=nothing: Weights to modify the parameter C for class i to class_weight[i]*C. The string \"balanced\"` is also accepted.\nmax_iter=-1: Limit the number of outer iterations in the solver. If -1 (default) then max_iter=100*n_samples.\nmulticlass_strategy=\"ovo\"\novo: OneVsOneClassifier\novr: OneVsRestClassifier\nnochange_steps=1000: Stop training if a 1e-3*tol difference isn't seen in nochange_steps steps.\nprobability=false: Enable or disable probability estimates.\nrandom_state=nothing: Seed for the random number generator.\nverbose=false: Sets logging level.\n\nOperations\n\npredict(mach, Xnew): return predictions of the target given   features Xnew having the same scitype as X above. Predictions   are class assignments. \npredict_proba(mach, Xnew): return predictions of the target given   features Xnew having the same scitype as X above. Predictions   are probabilistic, but uncalibrated.\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\nmodel: the trained model object created by the RAPIDS.jl package\n\nReport\n\nThe fields of report(mach) are:\n\nclasses_seen: list of target classes actually observed in training\nfeatures: the names of the features encountered in training.\n\nExamples\n\nusing RAPIDS\nusing MLJBase\n\nX = rand(100, 5)\ny = [repeat([0], 50)..., repeat([1], 50)...]\n\nmodel = SVC()\nmach = machine(model, X, y)\nfit!(mach)\npreds = predict(mach, X)\n\n\n\n\n\n","category":"type"},{"location":"cuml/#RAPIDS.LinearSVC","page":"cuMl","title":"RAPIDS.LinearSVC","text":"LinearSVC\n\nA model type for constructing a linear svc, based on cuML Classification Methods.jl, and implementing the MLJ model interface.\n\nFrom MLJ, the type can be imported using\n\nLinearSVC = @load LinearSVC pkg=cuML Classification Methods\n\nDo model = LinearSVC() to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in LinearSVC(penalty=...).\n\nLinearSVC is a wrapper for the RAPIDS LinearSVC.\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance model to data with     mach = machine(model, X, y)\n\nwhere\n\nX: any table or array of input features (eg, a DataFrame) whose columns   each have one of the following element scitypes: Continuous\ny: is the target, which can be any AbstractVector whose element   scitype is <:OrderedFactor or <:Multiclass; check the scitype   with scitype(y)\n\nTrain the machine using fit!(mach, rows=...).\n\nHyper-parameters\n\npenalty=\"l2: l1 (Lasso) or l2 (Ridge) penalty.\nloss=\"squared_hinge\": The loss term of the target function.\nfit_intercept=true: If true, the model tries to correct for the global mean of y.                        If false, the model expects that you have centered the data.\npenalized_intercept=true: When true, the bias term is treated the same way as other features.\nmax_iter=1000: Maximum number of iterations for the underlying solver.\nlinesearch_max_iter=1000: Maximum number of linesearch (inner loop) iterations for the underlying (QN) solver.\nlbfgs_memory=5: Number of vectors approximating the hessian for the underlying QN solver (l-bfgs).\nC=1.0: The constant scaling factor of the loss term in the target formula F(X, y) = penalty(X) + C * loss(X, y).\ngrad_tol=0.0001: The threshold on the gradient for the underlying QN solver.\nchange_tol=0.00001: The threshold on the function change for the underlying QN solver.\ntol=nothing: Tolerance for stopping criterion.\nprobabability=false: Enable or disable probability estimates.\nmulti_class=\"ovo\"\novo: OneVsOneClassifier\novr: OneVsRestClassifier\nverbose=false: Sets logging level.\n\nOperations\n\npredict(mach, Xnew): return predictions of the target given   features Xnew having the same scitype as X above. Predictions   are class assignments. \npredict_proba(mach, Xnew): return predictions of the target given   features Xnew having the same scitype as X above. Predictions   are probabilistic, but uncalibrated.\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\nmodel: the trained model object created by the RAPIDS.jl package\n\nReport\n\nThe fields of report(mach) are:\n\nclasses_seen: list of target classes actually observed in training\nfeatures: the names of the features encountered in training.\n\nExamples\n\nusing RAPIDS\nusing MLJBase\n\nX = rand(100, 5)\ny = [repeat([0], 50)..., repeat([1], 50)...]\n\nmodel = LinearSVC()\nmach = machine(model, X, y)\nfit!(mach)\npreds = predict(mach, X)\n\n\n\n\n\n","category":"type"},{"location":"cuml/#RAPIDS.KNeighborsClassifier","page":"cuMl","title":"RAPIDS.KNeighborsClassifier","text":"KNeighborsClassifier\n\nA model type for constructing a k neighbors classifier, based on cuML Classification Methods.jl, and implementing the MLJ model interface.\n\nFrom MLJ, the type can be imported using\n\nKNeighborsClassifier = @load KNeighborsClassifier pkg=cuML Classification Methods\n\nDo model = KNeighborsClassifier() to construct an instance with default hyper-parameters. Provide keyword arguments to override hyper-parameter defaults, as in KNeighborsClassifier(algorithm=...).\n\nKNeighborsClassifier is a wrapper for the RAPIDS K-Nearest Neighbors Classifier.\n\nTraining data\n\nIn MLJ or MLJBase, bind an instance model to data with     mach = machine(model, X, y)\n\nwhere\n\nX: any table or array of input features (eg, a DataFrame) whose columns   each have one of the following element scitypes: Continuous\ny: is the target, which can be any AbstractVector whose element   scitype is <:OrderedFactor or <:Multiclass; check the scitype   with scitype(y)\n\nTrain the machine using fit!(mach, rows=...).\n\nHyper-parameters\n\nn_neighbors=5: Default number of neighbors to query.\nalgorithm=\"brute\": Only one algorithm is currently supported.\nmetric=\"euclidean\": Distance metric to use.\nweights=\"uniform\": Sample weights to use. Currently, only the uniform strategy is supported.\nverbose=false: Sets logging level.\n\nOperations\n\npredict(mach, Xnew): return predictions of the target given   features Xnew having the same scitype as X above. Predictions   are class assignments. \npredict_proba(mach, Xnew): return predictions of the target given   features Xnew having the same scitype as X above. Predictions   are probabilistic, but uncalibrated.\n\nFitted parameters\n\nThe fields of fitted_params(mach) are:\n\nmodel: the trained model object created by the RAPIDS.jl package\n\nReport\n\nThe fields of report(mach) are:\n\nclasses_seen: list of target classes actually observed in training\nfeatures: the names of the features encountered in training.\n\nExamples\n\nusing RAPIDS\nusing MLJBase\n\nX = rand(100, 5)\ny = [repeat([0], 50)..., repeat([1], 50)...]\n\nmodel = KNeighborsClassifier()\nmach = machine(model, X, y)\nfit!(mach)\npreds = predict(mach, X)\n\n\n\n\n\n","category":"type"},{"location":"cuml/#Regression","page":"cuMl","title":"Regression","text":"","category":"section"},{"location":"cuml/","page":"cuMl","title":"cuMl","text":"    - `LinearRegression`\n    - `Ridge`\n    - `Lasso`\n    - `ElasticNet`\n    - `MBSGDRegressor`\n    - `RandomForestRegressor`\n    - `CD`\n    - `SVR`\n    - `LinearSVR`\n    - `KNeighborsRegressor`","category":"page"},{"location":"cuml/#Dimensionality-Reduction","page":"cuMl","title":"Dimensionality Reduction","text":"","category":"section"},{"location":"cuml/","page":"cuMl","title":"cuMl","text":"    - `PCA`\n    - `IncrementalPCA`\n    - `TruncatedSVD`\n    - `UMAP`\n    - `TSNE`\n    - `GaussianRandomProjection`","category":"page"},{"location":"cuml/#Time-Series","page":"cuMl","title":"Time Series","text":"","category":"section"},{"location":"cuml/","page":"cuMl","title":"cuMl","text":"    - `ExponentialSmoothing`\n    - `ARIMA`","category":"page"}]
}
