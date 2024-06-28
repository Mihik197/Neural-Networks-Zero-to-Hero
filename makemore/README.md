# Exercises from the lecture description


**E01**: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model?

The loss went down to 2.2485227584838867 in the trigram model (using a neural net). It definitely improved over our bigram model, which had a loss of 2.480508804321289



**E02**: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?

For bigram model:
training set loss: 2.4599320888519287
dev set loss: 2.4877490997314453
test set loss: 2.4755358695983887

For trigram model:
training set loss: 2.2483363151550293
dev set loss: 2.2581946849823
test set loss: 2.246502637863159

We notice that the loss is higher on our new training data sets, due to there being less data to train on. And slightly higher than that on the dev and test sets.



**E03**: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve?

Smoothness: 0.001 -> training set loss: 2.242676019668579, dev set loss: 2.239325761795044
  
Smoothness: 0.01 -> training set loss: 2.2591874599456787, dev set loss: 2.2474706172943115

Smoothness: 0.1 -> training set loss: 2.318784236907959, dev set loss: 2.2634406089782715

Smoothness: 1 -> training set loss: 2.5450072288513184, dev set loss: 2.3924319744110107


The loss on the training set increases as the smoothing strength increases. But the loss on the dev set is surprisingly lower than the loss on the training set



**E04**: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. Can you delete our use of F.one_hot in favor of simply indexing into rows of W?

Here's the line of code that let's us directly select the rows of W we want:
```
logits = W[xs[:, 0]] + W[xs[:, 1] + 27]
```



**E05**: look up and use F.cross_entropy instead. You should achieve the same result. Can you think of why we'd prefer to use F.cross_entropy instead?

Calculating loss using F.cross_entropy:
```
loss = F.cross_entropy(logits, ys)
```

It's much easier to directly pass in our input logits and target rather than having to calculate the 'counts' and 'probabilities' and then having to manually index into the 'probabilties' tensor.
