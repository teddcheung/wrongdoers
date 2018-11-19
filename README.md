# Moral Cost Project 2018 by Dr Daphna Buchsbaum, Teddy C.K. Cheung and Rachel Eng of CoCo Lab, Dept of Psychology, U of Toronto

// Real data from Tasimi and Wynn (2016): 
// baseline data (18/20, 18/20, 19/20, 16/20 confirmed by Arber;in paper: 71/80) 
var expData1b = _.flatten([repeat(2, function(){'personA'}), repeat(18, function(){'personB'})]) 
var expData2b = _.flatten([repeat(2, function(){'personA'}), repeat(18, function(){'personB'})]) 
var expData3b = _.flatten([repeat(1, function(){'personA'}), repeat(19, function(){'personB'})]) 
var expData4b = _.flatten([repeat(4, function(){'personA'}), repeat(16, function(){'personB'})])

var baselines = [expData1b, expData2b, expData3b, expData4b]
//var baselines = [expData4b]

// character-information data (4/20, 8/20, 8/20, 13/20) 
var expData1c = _.flatten([repeat(16, function(){'personA'}), repeat(4, function(){'personB'})]) 
var expData2c = _.flatten([repeat(12, function(){'personA'}), repeat(8, function(){'personB'})]) 
var expData3c = _.flatten([repeat(12, function(){'personA'}), repeat(8, function(){'personB'})]) 
var expData4c = _.flatten([repeat(7, function(){'personA'}), repeat(13, function(){'personB'})])

var testexps = [expData1c, expData2c, expData3c, expData4c]

// Define the Stickers Ratios of 1:2, 1:4, 1:8, 1:16 in face value 
// var stickerValsExp1 = {A: 1, B: 2} 
// var stickerValsExp2 = {A:1, B: 4} 
// var stickerValsExp3 = {A: 1, B: 8} 
// var stickerValsExp4 = {A: 1, B: 16}

// // Define the Stickers Ratios of 1:2, 1:4, 1:8, 1:16 in Log2 
var stickerValsExp1 = {A: Math.log2(1), B: Math.log2(2)} 
var stickerValsExp2 = {A: Math.log2(1), B: Math.log2(4)} 
var stickerValsExp3 = {A: Math.log2(1), B: Math.log2(8)} 
var stickerValsExp4 = {A: Math.log2(1), B: Math.log2(16)}

var allStickerVals = [stickerValsExp1, stickerValsExp2, stickerValsExp3, stickerValsExp4]

  var actions = ['personA', 'personB']

  var expectedUtility = function (action, meannessValues, stickerValues){
   
    var stickerProbs = (action === 'personB') ? [0.1, 0.9] : [0.9, 0.1]; 
    var meannessProbs = (action === 'personB') ? [0.1, 0.9] : [0.9, 0.1];    
    
    var reward = sum(map2(function(a, b){a*b}, stickerProbs, _.values(stickerValues)))
    var cost = sum(map2(function(a, b){a*b}, meannessProbs, _.values(meannessValues)))
    
    return (reward + cost)
  }
  
  print("expected utility person B " + expectedUtility('personB', {mean: 0,nice: 0}, {A: 1, B: 2}))
  print("expected utility person A " + expectedUtility('personA', {mean: 0,nice: 0}, {A: 1, B: 2}))

  var softMaxAgent = function(alpha, meannessValues, stickerValues) {
    return Infer({ 
      model() {

        var action = uniformDraw(actions);
        factor(alpha * expectedUtility(action, meannessValues, stickerValues));

        return action; 
      }, method: 'enumerate'});
  }; 

//examples testing some of the model results
var actionDist = softMaxAgent(1,{nice: 0, mean: -3}, {A: Math.log2(1), B: Math.log2(16)})
//var actionDist = softMaxAgent(1,{nice: 0, mean: -3}, {A: 1, B: 16})

viz.table(actionDist)

var actionDist = softMaxAgent(1,{nice: 0, mean: -3}, {A: Math.log2(1), B: Math.log2(2)})
//var actionDist = softMaxAgent(1,{nice: 0, mean: -3}, {A: 1, B: 2})

viz.table(actionDist)

  var observerFunction = function(actions, alpha, meannessValues, stickerValues) {
    map(function(action) {
      factor(softMaxAgent(alpha, meannessValues, stickerValues).score(action))
    }, actions)
  }

  //Outer model for inferring alpha from baseline conditions
  var outerModel = function(experiments)
  {
    var alpha = exponential(1)
    
    var meannessValues = {
    nice: 0, 
    mean: 0 //uniform({a: -16, b: 0}),
    }
 
    map2(function(actions,stickerValues) {
      observerFunction(actions, alpha, meannessValues, stickerValues)}, baselines, allStickerVals)
    
   return alpha
  }
  
 var alphaDist = Infer({method: 'MCMC', samples: 1000, burn: 100,  model: function() {return outerModel(baselines)}}); // the posterior
//var alphaDist = marginalize(paramDist, "alpha")
 viz.density(alphaDist)

//expected Value of alpha distribution
print(expectation(alphaDist))

//use expected alpha from baselines to infer meanness 
var meanAlpha = expectation(alphaDist)

 //Outer model for inferring meanness from test conditions
  var outerModel = function(actions)
  {
    var alpha = meanAlpha
    
    var meannessValues = {
    nice: 0, 
    mean: uniform({a: -16, b: 0})
    }
 
  map2(function(actions,stickerValues) {
      observerFunction(actions, alpha, meannessValues, stickerValues)}, testexps, allStickerVals)
       
    return meannessValues.mean
  }
  
   var meannessDist = Infer({method: 'MCMC', samples: 1000, burn: 100,  model: function() {return outerModel(expData2c)}}); // the posterior
viz.density(meannessDist)

//expected Value of alpha distribution
print(expectation(meannessDist))
