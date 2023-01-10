/* -*- tab-width:4 -*- */

use std::rc::Rc;
use crate::neural_network::NNModel;
use crate::neuron::{NeuronPrimType,Neuron,NNNeuron};

pub enum Optimizer {
    SGD, MomentumSDG
}

pub struct NNOptimizer<T>
where T: NeuronPrimType<T> {
    target: NNModel<T>,
    optimizer: Optimizer
}

impl<T> NNOptimizer<T>
where T: NeuronPrimType<T> {
    pub fn new(optimizer:Optimizer, model:&NNModel<T>) -> NNOptimizer<T> {
	NNOptimizer {
	    target: Rc::clone(model),
	    optimizer: optimizer
	}
    }
}
