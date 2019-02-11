# ---
# Title: "Machine Learning workshop: tree-based algorithms"
# author: "Masahiro Ryo @Freie Universitaet Berlin"
# date: "2019-02-12"
# output: html_document
# ---



## ----setwd---------------------------------------------------------------
setwd("C:/Users/masah/Dropbox/i_share/Event & conference/20190212_UFZ/workshop")
getwd()

## ----packages, results="hide", warning=FALSE, message=FALSE--------------
package.list = c("party", "mlr","parallelMap", "rpart", "rpart.plot","mmpf")
tmp.install = which(lapply(package.list, require, character.only = TRUE)==FALSE)
if(length(tmp.install)>0) install.packages(package.list[tmp.install], repos = "http://cran.us.r-project.org")
lapply(package.list, require, character.only = TRUE)

## ----read_df-------------------------------------------------------------
df = read.csv("data_example.csv")
head(df, 3)

## ----formula-------------------------------------------------------------
# creating formula
formula.1 = as.formula(paste("y1", paste(colnames(df)[2:length(colnames(df))], collapse=" + "), sep=" ~ ")) 
print(formula.1)

## ----cart----------------------------------------------------------------
# regression
cart.1 = rpart(formula.1, data=df, method="anova",control=rpart.control(minsplit=10, cp=0.001)) 

## ----cart_plot-----------------------------------------------------------
prp(cart.1) # plot with rpart.plot package

## ----cart_cv-------------------------------------------------------------
plotcp(cart.1)	#plot cross-validation results
rsq.rpart(cart.1)	#plot approximate R-squared and relative error for different splits (2 plots). labels are only appropriate for the "anova" method.
cp.best = cart.1$cptable[which(cart.1$cptable[,"xerror"]==min(cart.1$cptable[,"xerror"])),"CP"]

## ----cart_best-----------------------------------------------------------
# regression (y1) and classification (y2)
cart.1.best = rpart(formula.1, data=df, method="anova",control=rpart.control(minsplit=10, cp=cp.best)) 

par(mfrow=c(1,2))
prp(cart.1.best)
prp(cart.1)

## ----ctree---------------------------------------------------------------
ctree.1 = ctree(formula.1, df, control = ctree_control(testtype = "Bonferroni", mincriterion = 0.95, minsplit = 10, minbucket = 7))
plot(ctree.1)

## ----ctree_p-value-------------------------------------------------------
ctree.2 = ctree(formula.1, df, control = ctree_control(testtype = "Univariate", mincriterion = 0.95, minsplit = 10, minbucket = 7))
plot(ctree.2)

## ----parallel------------------------------------------------------------
parallelStartSocket(3)

## ----cforest_mlr---------------------------------------------------------
regr.task = makeRegrTask(data = df, target = "y1")
regr.learner.cforest = makeLearner(cl="regr.cforest", predict.type = "response")
model.cforest = train(regr.learner.cforest, regr.task)

## ----cforest_eval--------------------------------------------------------
pred.cforest = predict(model.cforest, task = regr.task)
listMeasures(regr.task)
performance(pred.cforest, measures = list(mse, rsq))

## ----cforest_cv, include=FALSE-------------------------------------------
rdesc = makeResampleDesc("CV", iters = 3)

# Calculate the performance
cv.cforest = resample("regr.cforest", regr.task, rdesc, measures = list(mse, rsq), extract = getFeatureImportance)

## ----cforest_cv_result---------------------------------------------------
cv.cforest$aggr
cv.cforest$measures.test

## ----cforest_tune, include=FALSE-----------------------------------------
ps = makeParamSet(
  makeIntegerParam("ntree", lower = 10, upper = 500),
  makeIntegerParam("mtry", lower =  1, upper = 10)
)

ctrl = makeTuneControlGrid()
tune.cforest = tuneParams("regr.cforest", task = regr.task, resampling = rdesc, par.set = ps, control = ctrl)

## ----cforest_best--------------------------------------------------------
model.cforest.best = train(setHyperPars(makeLearner("regr.cforest"), par.vals = tune.cforest$x),regr.task)

## ----cforest_tunemap-----------------------------------------------------
plotHyperParsEffect(generateHyperParsEffectData(tune.cforest), x = "ntree", y = "mtry", z = "mse.test.mean",
  plot.type = "heatmap")


## ----cforest_importance--------------------------------------------------
vimp = generateFilterValuesData(regr.task, method = "cforest.importance")
plotFilterValues(vimp) 

## ----cforest_selection---------------------------------------------------
lrn = makeFilterWrapper(learner = "regr.cforest", 
  fw.method = "cforest.importance", fw.abs = 5)
r = resample(learner = lrn, task = regr.task, resampling = rdesc, show.info = FALSE, models = TRUE)
r$aggr

## ----cforest_pdplot_x9---------------------------------------------------
pd = generatePartialDependenceData(model.cforest.best, regr.task, "x9", individual = F)
plotPartialDependence(pd)

## ----cforest_pdplot_all--------------------------------------------------
pd.all = generatePartialDependenceData(model.cforest.best, regr.task, individual = F)
plotPartialDependence(pd.all)

## ----cforest_ice_x9------------------------------------------------------
ice = generatePartialDependenceData(model.cforest.best, regr.task, "x9", individual = T)
plotPartialDependence(ice)

## ----cforest_ice_all-----------------------------------------------------
ice.all = generatePartialDependenceData(model.cforest.best, regr.task, individual = T)
plotPartialDependence(ice.all)

## ----end-----------------------------------------------------------------
parallelStop()

