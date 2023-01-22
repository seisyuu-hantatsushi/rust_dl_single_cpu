
use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NNNeuron, nn_neuron_new, nn_neuron_constant};

impl<T> SynapseNode<T>
where T:num::Float + num::FromPrimitive + num::pow::Pow<T, Output = T> + Clone + fmt::Display {
    fn softmax(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
	
    }
}
