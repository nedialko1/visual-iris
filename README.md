# visual-iris
A companion set of eyes-centric (visually enriched) python codes toward explainable AI (XAI) and other didactic purposes 

**How to use toward best results**

The repo is organized in the following way:
* Its **root folder** contains all you need to get started - e.g. the minimum *requirements* file and a fully functional 
*PyTorch* script that implements and then runs (inference, training etc) the well-known **baseline** version of
the *Iris Flowers* classifier MLP network model. This baseline model is then examined under the magnifying glass of
*two* possible approaches toward the model's better understanding and further refinement.   
These approaches have dedicated subfolders as follows:
Note: Representative figure examples provide a visual intuition as to what to expect from these approaches.

---

* da_Visual_Iris
![Setosa_Baseline.PNG](./images/Setosa_Baseline.PNG)

This approach can be called "Open the Black Box". It relies on intuitive visual unfolding of a given connectionist model.
The relative simplicity of the *Iris Flowers* classifier MLP network provides for an easy entry into the world of
turning connectionist models into more *explainable* AI (XAI). Clearly the approach is not yet readily scalable to 'production'-
level AI - i.e. to readily translate to visualize NN models with hundreds of layers, thousands of connections to single model neurons,
and billions of parameters. However, it is also easily provable that a rather similar approach would work *also* at 'production'
scale, provided it is complemented with ways to abstract out various levels of model architecture complexity.
E.g. applying the concept of *Russian dolls* the models could still be presented in a quite intuitively explainable visual form 
provided the latter is more *structured* - and consisting of (going/zooming-in from the largest to progressively lower scales):
systems, subsystems, modules, ... all the way to individual layers and neurons - just as illustrated here.

Last but not least, these "Visual Iris" scripts offer a key extra functionality. Namely the possibility to visually explore *Firing Thresholds*.
Lately, it has been observed that 'production' scale models inference and especially training on the conventionally used superclusters and GPU's
consume more than a fair share of Energy. Put this way one sees that AI not only is not yet resolving global warming, but is actually contributing to it!
Why this is currently the state with (already conventional) AI/ML?
Because currently **all** units and connections in an NN model loaded in VRAM consume electrical power *regardless* of whether they contribute or not any 
significant fraction of the model's behavior (e.g. the prediction of the next word in the sentence, or the class of iris flower at hand)!
Hence, heads are turning to more biologically inspired intelligence, and in particular information storage in the brain itself - where encoding is quite sparse - 
only a handful of neurons in a given subsytem actually fire *action potentials* (AP, also known as *spikes*).
Spiking NN (SNN) models hold also the promise of adding a crucial information-content dimension to rate coding - namely the actual spike *timing*.
It has been shown in very influential and now classic neuroscience research that in many very practical situations a first spike is all it takes to 
classify a visual object - making the brain's solution not only less energy consuming but also way faster. 
Now, imagine if a similar to the living brain's paradigms was to be put inside modern fast computers...

Toward such a purpose one may wonder what would be a way to convert existing models into SNN ones.
And the answer to such questioning may be simpler than we think.
Turning existing model neurons in *spiking* ones may be as simple as introducing just another type of "activation function*.
Namely, turn off all model neurons whose activation is below a preset **firing threshold** parameter.
A proof of concept model is presented here in a very visual form - providing the user with sliders to set the firing threshold 
for any of the NN model's layers. (Do play with these sliders :)

---

* Statistical_Iris
  ![stats_iris_2D.png](./images/stats_iris_2D.png)

The relative simplicity of the baseline *Iris Flowers* classifier MLP network may actually be a bit of an illusion.  
Its 4 - 8 - 9 - 3  MLP model structure still requires more than 150 (!) weight and bias parameters.
If we compare this to classical statistics methods, we shall see that good old statistics 
would achieve similar performance in the same 3 classes of iris flowers discrimination using less than 10 (!) parameters, 
see Principal Component Analysis (PCA), Fisher Linear Discrimination (FLD) etc.
Moreover, the optimal values of these about 10 parameters would be available in closed form and not require *any* training whatsoever!

Hence, this second alternative approach focuses on harnessing powerful statistics methods. The latter do not have to be considered 
mere rivals in the quest for explainable AI. They can readily *complement*  connectionist modeling and greatly contribute to their analysis,
structural optimisation and especially their *explainability*.

To illustrate the above potential, PCA and FLD are used here to:
  - suggest an alternative  4 - 4 - 3 - 3  MLP model structure that now has **three times** less parameters and is just as good a clasifier as the baseline.
  - reliably demonstrate - on a rather intuitively accessible NN model, that over-parameterised models may achieve acceptable performance, 
  while their hidden layers *still fail* to capture almost *any* reasonable problem structure  


**General Notes:**

- This little project stems from a long-standing desire toward ever more *explainable* XAI and 
to generally make connectionist models intuitively clearer and more accessible to the only real intelligence there is so far - 
that of human beings.

- This set of python codes is planned as a companion to a Medium.com article with the working title 
"Frugal Explainable AI/ML Model Optimization - a Hands-On Introduction".

- Once upon a time key human projects (e.g. the pioneer launches of spacecraft) were driven by computers programmed in Fortran.
No so long ago seminal and authoritative Fortran libraries were translated in "C" or "C++" using automatic translations.
Time has come to delegate some of the coding in implementations to AI agency. This applies also to this repo's code.
However, all the underlying ideas and algorithms are 100% original and human. Coding task lists and any other 'recipes' were prescribed
verbatim to the AI coder whenever applicable, and the result of AI's work was rigorously tested, validated and corrected where it was necessary. 
