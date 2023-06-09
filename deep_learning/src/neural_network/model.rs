/* -*- tab-width:4 -*- */

use std::fs;
use std::io::Write;
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashMap;

use linear_transform::tensor::Tensor;
use crate::neuron::{NeuronPrimType,Neuron,NNNeuron};
use crate::synapse::SynapseNode;
use crate::neural_network::NeuralNetwork;
use crate::neural_network::layer::NNLayer;

#[derive(Copy,Clone,Debug,PartialEq)]
pub enum MLPActivator {
	Sigmoid,
	ReLU,
}

pub struct TwoLayerNet<T>
where T: NeuronPrimType<T> {
	ll1: NNLayer<T>,
	ll2: NNLayer<T>,
}

pub type NNTwoLayerNet<T> = RefCell<TwoLayerNet<T>>;

pub struct MLP<T>
where T: NeuronPrimType<T> {
	lls: Vec<NNLayer<T>>,
	activator: MLPActivator
}

pub type NNMLP<T> = RefCell<MLP<T>>;

pub enum EModel<T>
where T: NeuronPrimType<T> {
	TwoLayerNet(NNTwoLayerNet<T>),
	MLP(NNMLP<T>)
}

type NNSynapseNodeHashMap<T> = HashMap<* const RefCell<SynapseNode<T>>, Rc<RefCell<SynapseNode<T>>>>;

pub struct Model<T>
where T: NeuronPrimType<T> {
	synapse_nodes: NNSynapseNodeHashMap<T>,
	model: EModel<T>
}

pub type NNModel<T> = Rc<Model<T>>;

impl<T> NeuralNetwork<T>
where T:NeuronPrimType<T> {

	pub fn create_two_layer_net_model(&mut self,
									  name: &str,
									  hidden_size:usize,
									  out_size:usize) -> NNModel<T> {
		let ll1 = self.create_linear_layer(&(name.to_string() + "_LL1"),
										   hidden_size,
										   true);
		let ll2 = self.create_linear_layer(&(name.to_string()+"_LL2"),
										   out_size,
										   true);
		Rc::new(Model {
			synapse_nodes: HashMap::new(),
			model: EModel::TwoLayerNet(
				RefCell::new(TwoLayerNet {
					ll1: ll1,
					ll2: ll2
				}))
		})
	}

	pub fn create_mlp_model(&mut self,
							name: &str,
							out_sizes:&[usize],
							activator:MLPActivator) -> NNModel<T> {

		let lls:Vec<NNLayer<T>> = (0..out_sizes.len()).map(|i| {
			let label = name.to_string() + "_LL" + &i.to_string();
			self.create_linear_layer(&label, out_sizes[i], true)
		}).collect();

		Rc::new(Model {
			synapse_nodes: HashMap::new(),
			model: EModel::MLP(
				RefCell::new(MLP {
					lls:lls,
					activator: activator
				}))})
	}

	fn two_layer_net_set_inputs(&mut self,
								sns_map:&mut NNSynapseNodeHashMap<T>,
								model:&mut TwoLayerNet<T>,
								inputs:Vec<NNNeuron<T>>) -> Vec<NNNeuron<T>> {

		let ll1_outputs = self.layer_set_inputs(&mut model.ll1, inputs);
		let activetor   = self.sigmoid(Rc::clone(&ll1_outputs[0]));
		let ll2_outputs = self.layer_set_inputs(&mut model.ll2, vec![Rc::clone(&activetor)]);

		for nn in vec![ll1_outputs, vec![activetor], ll2_outputs.clone()].concat().iter() {
			let sns = self.get_linked_synapses(0, nn);
			for sn in sns.iter() {
				sns_map.insert(Rc::as_ptr(sn),Rc::clone(sn));
			}
		}

		ll2_outputs
	}

	fn two_layer_net_set_weights_and_inputs(&mut self,
											sns_map:&mut NNSynapseNodeHashMap<T>,
											model:&mut TwoLayerNet<T>,
											weights: &[(Tensor<T>,Option<Tensor<T>>)],
											inputs:Vec<NNNeuron<T>>) -> Vec<NNNeuron<T>> {

		let ll1_outputs = self.layer_set_weights_and_inputs(&mut model.ll1, weights[0].clone(), inputs);
		let activetor     = self.sigmoid(Rc::clone(&ll1_outputs[0]));
		let ll2_outputs = self.layer_set_weights_and_inputs(&mut model.ll2, weights[1].clone(), vec![Rc::clone(&activetor)]);

		for nn in vec![ll1_outputs, vec![activetor], ll2_outputs.clone()].concat().iter() {
			let sns = self.get_linked_synapses(0, nn);
			for sn in sns.iter() {
				sns_map.insert(Rc::as_ptr(sn),Rc::clone(sn));
			}
		}

		ll2_outputs
	}

	fn mlp_set_inputs(&mut self,
					  sns_map:&mut NNSynapseNodeHashMap<T>,
					  model:&mut MLP<T>,
					  inputs:Vec<NNNeuron<T>>) -> Vec<NNNeuron<T>> {
		let mut layer_inputs:Vec<NNNeuron<T>> = inputs;
		let num_of_layers = model.lls.len();
		let mut layer_outputs:Vec<Vec<NNNeuron<T>>> = vec!();

		for i in 0..(num_of_layers-1) {
			let l_outputs = self.layer_set_inputs(&mut model.lls[i], layer_inputs);
			let activator = if model.activator == MLPActivator::ReLU
			{
				self.relu(Rc::clone(&l_outputs[0]))
			}
			else
			{
				self.sigmoid(Rc::clone(&l_outputs[0]))
			};
			layer_inputs = vec![activator];
			layer_outputs.push(l_outputs);
			layer_outputs.push(layer_inputs.clone());
		}

		let outputs = self.layer_set_inputs(&mut model.lls[num_of_layers-1], layer_inputs);
		layer_outputs.push(outputs.clone());

		for nn in layer_outputs.concat().iter() {
			let sns = self.get_linked_synapses(0, nn);
			for sn in sns.iter() {
				sns_map.insert(Rc::as_ptr(sn),Rc::clone(sn));
			}
		}

		outputs
	}

	fn mlp_net_set_weights_and_inputs(&mut self,
									  sns_map:&mut NNSynapseNodeHashMap<T>,
									  model:&mut MLP<T>,
									  weights: &[(Tensor<T>,Option<Tensor<T>>)],
									  inputs:Vec<NNNeuron<T>>) -> Vec<NNNeuron<T>> {
		let mut layer_inputs:Vec<NNNeuron<T>> = inputs;
		let num_of_layers = model.lls.len();
		let mut layer_outputs:Vec<Vec<NNNeuron<T>>> = vec!();

		for i in 0..(num_of_layers-1) {
			let l_outputs = self.layer_set_weights_and_inputs(&mut model.lls[i],
															  weights[i].clone(),
															  layer_inputs);
			let label = "affine_".to_string() + &i.to_string();
			l_outputs[0].borrow_mut().rename(&label);
			layer_inputs = vec![self.sigmoid(Rc::clone(&l_outputs[0]))];
			layer_outputs.push(l_outputs);
			layer_outputs.push(layer_inputs.clone());
		}

		let outputs = self.layer_set_weights_and_inputs(&mut model.lls[num_of_layers-1],
														weights[num_of_layers-1].clone(),
														layer_inputs);
		let label = "affine_".to_string() + &(num_of_layers-1).to_string();
		outputs[0].borrow_mut().rename(&label);
		layer_outputs.push(outputs.clone());

		for nn in layer_outputs.concat().iter() {
			let sns = self.get_linked_synapses(0, nn);
			for sn in sns.iter() {
				sns_map.insert(Rc::as_ptr(sn),Rc::clone(sn));
			}
		}

		outputs
	}

	pub fn model_set_inputs(&mut self,
							nnmodel:&mut NNModel<T>,
							inputs:Vec<NNNeuron<T>>) -> Vec<NNNeuron<T>> {
		let model = Rc::get_mut(nnmodel).unwrap_or_else(|| panic!("not safe to mutate"));
		match &model.model {
			EModel::TwoLayerNet(m) => {
				self.two_layer_net_set_inputs(&mut model.synapse_nodes, &mut m.borrow_mut(), inputs)
			},
			EModel::MLP(m) => {
				self.mlp_set_inputs(&mut model.synapse_nodes, &mut m.borrow_mut(), inputs)
			}
		}
	}

	pub fn model_set_weight_and_inputs(&mut self,
									   nnmodel:&mut NNModel<T>,
									   weights:&[(Tensor<T>,Option<Tensor<T>>)],
									   inputs:Vec<NNNeuron<T>>) -> Vec<NNNeuron<T>> {
		let model = Rc::get_mut(nnmodel).unwrap_or_else(|| panic!("not safe to mutate"));
		match &model.model {
			EModel::TwoLayerNet(m) => {
				self.two_layer_net_set_weights_and_inputs(&mut model.synapse_nodes, &mut m.borrow_mut(), weights, inputs)
			},
			EModel::MLP(m) => {
				self.mlp_net_set_weights_and_inputs(&mut model.synapse_nodes, &mut m.borrow_mut(), weights, inputs)
			}
		}
	}
}

impl<T> Model<T>
where T:NeuronPrimType<T> {

	pub fn get_params(&self) -> Vec<NNNeuron<T>> {
		match &self.model {
			EModel::TwoLayerNet(m) => {
				vec![m.borrow().ll1.get_params(),m.borrow().ll2.get_params()].concat()
			},
			EModel::MLP(m) => {
				m.borrow().lls.iter().map(|ll| ll.get_params()).collect::<Vec<Vec<NNNeuron<T>>>>().concat()
			}
		}
	}
	pub fn make_dot_graph(&self, file_name:&str) -> Result<(),String> {
		let mut ns:HashMap<* const RefCell<Neuron<T>>, NNNeuron<T>> = HashMap::new();
		for sn in self.synapse_nodes.values() {
			for ni in sn.borrow().input_neurons().iter() {
				ns.insert(Rc::as_ptr(ni),Rc::clone(ni));
			}
			for no in sn.borrow().output_neurons().iter() {
				ns.insert(Rc::as_ptr(no),Rc::clone(no));
			}
		}
		let mut output = "digraph g {\n".to_string();
		for n in ns.values() {
			let nref = n.borrow();
			let id_str = format!("{:p}", Rc::as_ptr(&n));
			let color = if nref.is_constant() {
				"seagreen4"
			}
			else {
				"orange"
			};
			let mut shape_str = "[".to_string();
			for s in nref.ref_signal().shape().iter() {
				shape_str = shape_str + &s.to_string() + ",";
			}
			shape_str = shape_str + "]";
			let label = nref.name().to_string()+" "+&shape_str+"\n"+"id="+&id_str;
			output = output.to_string() + "\"" + &id_str + "\"" + " [label=\"" + &label + "\", color=" + &color + ", style=filled]" + "\n";
		}

		for sn in self.synapse_nodes.values() {
			let sn_ref = sn.borrow();
			let label = sn_ref.name().to_string();
			let id_str = format!("{:p}", Rc::as_ptr(&sn));
			output = output.to_string() + "\"" + &id_str + "\"" + " [label=\"" + &label + "\", color=lightblue, style=filled, shape=box ]" + "\n";
			for ni in sn_ref.inputs().iter() {
				let niid_str = format!("{:p}",Rc::as_ptr(&ni));
				output = output.to_string() + "\"" + &niid_str.to_string() + "\"->\"" + &id_str + "\"\n"
			}

			for no in sn_ref.outputs().iter() {
				let noid_str = format!("{:p}", Rc::as_ptr(&no));
				output = output.to_string() + "\"" + &id_str + "\"->\"" + &noid_str + "\"\n"
			}
		}

		output = output + "}";
		let mut ofs = fs::File::create(file_name).unwrap();
		if let Err(e) = ofs.write_all(output.as_bytes()){
			println!("{}",e);
			return Err(e.to_string());
		}
		Ok(())
	}
}
