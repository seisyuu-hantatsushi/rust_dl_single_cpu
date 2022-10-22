use std::{fmt,fs,ptr};
use std::io::Write;
use std::ops::Deref;
use std::cell::{Ref,RefCell};
use std::rc::Rc;
use std::collections::HashMap;
use linear_transform::tensor::tensor_base::Tensor;
use num;

use openssl::sha::sha512;

use crate::neuron::Neuron;
use crate::synapse::Synapse;

struct SynapseNode<T>
where T:num::Float + num::FromPrimitive + num::pow::Pow<T, Output = T> + Clone + std::fmt::Debug {
    hashed_id: [u8;64],
    name: String,
    synapse: Synapse<T>,
    generation: usize,
    inputs: Vec<Rc<RefCell<Neuron<T>>>>,
    outputs: Vec<Rc<RefCell<Neuron<T>>>>,
}

impl<T> fmt::Display for SynapseNode<T>
where T: fmt::Display + std::fmt::Debug + Clone + num::Float + num::pow::Pow<T, Output = T> + num::FromPrimitive {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	let mut disp = format!("SynapseNode. name:{}\n",self.name);
	disp = format!("generation {}", self.generation);
	write!(f,"{}",disp)
    }
}

impl<T> SynapseNode<T>
where T:num::Float + num::FromPrimitive + num::pow::Pow<T, Output = T> + Clone + std::fmt::Debug {

    pub fn new(prevs: Vec<Rc<RefCell<SynapseNode<T>>>>,
	       name: &str,
	       inputs: Vec<Rc<RefCell<Neuron<T>>>>,
	       outputs: Vec<Rc<RefCell<Neuron<T>>>>,
	       forward : fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
	       backward: fn (inputs: &Vec<&Tensor<T>>,grad:&Tensor<T>) -> Vec<Tensor<T>>) -> SynapseNode<T> {

	let s = Synapse::<T>::new(forward, backward);
	let generation:usize = prevs.iter().fold(0,|max,sn| {
	    if max < sn.borrow().generation + 1 {
		sn.borrow().generation + 1
	    }
	    else {
		max
	    }
	});

	SynapseNode {
	    hashed_id: sha512(name.as_bytes()),
	    name: name.to_string(),
	    synapse :s,
	    generation,
	    inputs,
	    outputs
	}
    }

    pub fn from_synapse(prevs: Vec<Rc<RefCell<SynapseNode<T>>>>,
			name: &str,
			inputs: Vec<Rc<RefCell<Neuron<T>>>>,
			outputs: Vec<Rc<RefCell<Neuron<T>>>>,
			synapse: Synapse<T> ) -> SynapseNode<T> {

	let generation:usize = prevs.iter().fold(0,|max,sn| {
	    if max < sn.borrow().generation + 1 {
		sn.borrow().generation + 1
	    }
	    else {
		max
	    }
	});

	SynapseNode {
	    hashed_id: sha512(("synapse_node".to_string() + name).as_bytes()),
	    name: name.to_string(),
	    synapse,
	    generation,
	    inputs,
	    outputs
	}
    }

    pub fn generation(&self) -> usize {
	self.generation
    }

    pub fn name(&self) -> &str {
	&self.name
    }

    pub fn id(&self) -> &[u8] {
	&self.hashed_id
    }

    pub fn inputs(&self) -> &Vec<Rc<RefCell<Neuron<T>>>> {
	&self.inputs
    }

    pub fn outputs(&self) -> &Vec<Rc<RefCell<Neuron<T>>>> {
	&self.outputs
    }

    pub fn forward_prop(&mut self) -> () {
	let inputs:Vec<Rc<RefCell<Tensor<T>>>> = self.inputs.iter().map(|n| n.borrow().get_signal()).collect();
	let outputs = self.synapse.forward(&inputs);
	for (os,on) in self.outputs.iter().zip(outputs.iter()) {
	    os.borrow_mut().assign(on.clone());
	}
    }

    pub fn backward_prop(&mut self) -> () {
	let grads:Vec<Rc<RefCell<Tensor<T>>>> =
	    self.outputs.iter().map(|output| {
		if None == output.borrow_mut().grad() {
		    output.borrow_mut().set_grad(Rc::new(RefCell::new(Tensor::<T>::one(&[1,1]))));
		}
		if let Some(grad) = output.borrow_mut().grad() {
		    return Rc::new(RefCell::new(grad));
		}
		else {
		    panic!();
		}
	    }).collect();
	let input_tensors = self.inputs.iter().map(|n| n.borrow().get_signal()).collect();
	let outputs = self.synapse.backward(&input_tensors, Rc::clone(&grads[0]));
	for (input, output) in self.inputs.iter().zip(outputs.into_iter()) {
	    let grad:Rc<RefCell<Tensor<T>>> = if let Some(g) = input.borrow().grad() {
		let grad = g + output;
		Rc::new(RefCell::new(grad))
	    }
	    else {
		Rc::new(RefCell::new(output.to_owned()))
	    };
	    input.borrow_mut().set_grad(grad);
	}
    }
}

struct NeuralNetwork<T>
where T:num::Float + num::FromPrimitive + num::pow::Pow<T, Output = T> + Clone + std::fmt::Debug {
    neurons:  HashMap<String, Rc<RefCell<Neuron<T>>>>,
    synapse_nodes: HashMap<String, Rc<RefCell<SynapseNode<T>>>>,
    input_neurons: Vec<Rc<RefCell<Neuron<T>>>>,
    generation_table: Vec<Vec<Rc<RefCell<SynapseNode<T>>>>>
}

impl<T> NeuralNetwork<T>
where T:num::Float + num::FromPrimitive + num::pow::Pow<T, Output = T> + Clone + std::fmt::Debug{

    pub fn new() -> NeuralNetwork<T> {
	NeuralNetwork {
	    neurons: HashMap::new(),
	    synapse_nodes: HashMap::new(),
	    input_neurons: Vec::new(),
	    generation_table: Vec::new()
	}
    }

    pub fn create_neuron(&mut self, name:&str, input: bool) -> Rc<RefCell<Neuron<T>>> {
	let n = Rc::new(RefCell::new(Neuron::<T>::create(name, Tensor::<T>::zero(&[1,1]))));
	self.neurons.insert(name.to_string(), Rc::clone(&n));
	if input {
	    self.input_neurons.push(Rc::clone(&n))
	}
	n
    }

    pub fn add_neuron(&mut self, n:Neuron<T>) -> Rc<RefCell<Neuron<T>>> {
	let name = n.name().to_string();

	if self.neurons.contains_key(&name) {
	    panic!("already contains label")
	}

	let rcn = Rc::new(RefCell::new(n));
	self.neurons.insert(name, Rc::clone(&rcn));
	rcn
    }

    pub fn create_synapse(&mut self,
			  prevs : Vec<&str>,
			  name: &str,
			  inputs: Vec<&str>,
			  outputs: Vec<&str>,
			  forward : fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
			  backward: fn (inputs: &Vec<&Tensor<T>>, grad:&Tensor<T>) -> Vec<Tensor<T>>) {
	let input_neurons:Vec<Rc<RefCell<Neuron<T>>>> =
	    inputs.iter().map(|label| Rc::clone(&self.neurons[*label])).collect();
	let output_neurons:Vec<Rc<RefCell<Neuron<T>>>> =
	    outputs.iter().map(|label| Rc::clone(&self.neurons[*label])).collect();

	let prev_nodes = prevs.iter().map(|label| { Rc::clone(&self.synapse_nodes[*label]) }).collect();

	let rsn = Rc::new(RefCell::new(SynapseNode::new(prev_nodes, name, input_neurons, output_neurons, forward, backward)));

	let g = rsn.borrow().generation();
	if self.generation_table.len() < g+1 {
	    self.generation_table.resize(g+1, Vec::new());
	}
	self.generation_table[g].push(Rc::clone(&rsn));
	self.synapse_nodes.insert(name.to_string(), Rc::clone(&rsn));
	rsn.borrow_mut().forward_prop();
    }

    pub fn add_synapse(&mut self,
		       prevs : Vec<&str>,
		       name: &str,
		       inputs: Vec<&str>,
		       outputs: Vec<&str>,
		       synapse: Synapse<T>){
	let input_neurons:Vec<Rc<RefCell<Neuron<T>>>> =
	    inputs.iter().map(|label| {
		if self.neurons.contains_key(*label) {
		    Rc::clone(&self.neurons[*label])
		}
		else {
		    panic!("not found neuron label {}", label);
		}
	    }).collect();
	let output_neurons:Vec<Rc<RefCell<Neuron<T>>>> =
	    outputs.iter().map(|label| {
		if self.neurons.contains_key(*label) {
		    Rc::clone(&self.neurons[*label])
		}
		else {
		    let n = Rc::new(RefCell::new(Neuron::<T>::create(label, Tensor::<T>::zero(&[1,1]))));
		    self.neurons.insert(label.to_string(), Rc::clone(&n));
		    n
		}
	    }).collect();
	let prev_nodes = prevs.iter().map(|label| {
	    if self.synapse_nodes.contains_key(*label) {
		Rc::clone(&self.synapse_nodes[*label])
	    }
	    else {
		panic!("not found synapse node label {}", label);
	    }
	}).collect();
	let rsn = Rc::new(RefCell::new(SynapseNode::from_synapse(prev_nodes, name, input_neurons, output_neurons, synapse)));
	let g = rsn.borrow().generation();

	if self.generation_table.len() < g+1 {
	    self.generation_table.resize(g+1, Vec::new());
	}
	self.generation_table[g].push(Rc::clone(&rsn));
	self.synapse_nodes.insert(name.to_string(), Rc::clone(&rsn));

	rsn.borrow_mut().forward_prop();
    }

    pub fn ref_neuron(&self, name:&str) -> &Rc<RefCell<Neuron<T>>> {
	&self.neurons[name]
    }

    pub fn ref_synapse_node(&self, name:&str) -> &Rc<RefCell<SynapseNode<T>>> {
	&self.synapse_nodes[name]
    }

    pub fn forward_prop(&mut self, inputs: Vec<Tensor<T>>) {

	for (i,n) in inputs.iter().zip(self.input_neurons.iter()) {
	    n.borrow_mut().assign(i.clone())
	}

	for g in self.generation_table.iter() {
	    for s in g.iter() {
		s.borrow_mut().forward_prop()
	    }
	}
    }

    pub fn backward_prop(&mut self, remain_grad: bool){
	for rg in self.generation_table.iter().rev() {
	    for s in rg.iter() {
		//println!("{}", s.borrow().name());
		s.borrow_mut().backward_prop()
	    }
	}

	if !remain_grad {
	    for g in 1..self.generation_table.len() {
		for sn in self.generation_table[g].iter() {
		    for ni in  sn.borrow().inputs.iter() {
			ni.borrow_mut().clear_grad();
		    }
		}
	    }
	}
    }

    pub fn clear_grad(&mut self){
	for n in self.neurons.values() {
	    n.borrow_mut().clear_grad()
	}
    }

    pub fn gen_dot_graph(&mut self, file: &str) -> (){
	let mut output = "digraph g {\n".to_string();
	for n in self.neurons.values() {
	    let nref = n.borrow();
	    let id_ref = nref.id();
	    let mut id:[u8;16] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
	    unsafe {
		std::ptr::copy(id_ref.as_ptr(), id.as_mut_ptr(), 16);
	    }
	    let id_u128 = u128::from_be_bytes(id);
	    output = output.to_string() + &id_u128.to_string() + " [label=\"" + nref.name() + "\", color=orange, style=filled]" + "\n";
	}
	for sn in self.synapse_nodes.values() {
	    let sn_ref = sn.borrow();
	    let id_ref = sn_ref.id();
	    let name = sn_ref.name();
	    let mut id:[u8;16] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
	    unsafe {
		std::ptr::copy(id_ref.as_ptr(), id.as_mut_ptr(), 16);
	    }
	    let id_u128 = u128::from_be_bytes(id);
	    output = output.to_string() + &id_u128.to_string() + " [label=\"" + name + "\", color=lightblue, style=filled, shape=box ]" + "\n";
	    for ni in sn_ref.inputs().iter() {
		let ni_ref = ni.borrow();
		let niid_ref = ni_ref.id();
		let mut niid:[u8;16] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
		unsafe {
		    std::ptr::copy(niid_ref.as_ptr(), niid.as_mut_ptr(), 16);
		}
		let niid_u128 = u128::from_be_bytes(niid);
		output = output.to_string() + &niid_u128.to_string() + "->" + &id_u128.to_string() + "\n"
	    }

	    for no in sn_ref.outputs().iter() {
		let no_ref = no.borrow();
		let noid_ref = no_ref.id();
		let mut noid:[u8;16] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
		unsafe {
		    std::ptr::copy(noid_ref.as_ptr(), noid.as_mut_ptr(), 16);
		}
		let noid_u128 = u128::from_be_bytes(noid);
		output = output.to_string() + &id_u128.to_string() + "->" + &noid_u128.to_string() + "\n"
	    }
	}

	output = output + "}";
	let mut ofs = fs::File::create(file).unwrap();
	ofs.write_all(output.as_bytes());

    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_test_add() {
	let mut nn = NeuralNetwork::<f32>::new();
	let n1 = Tensor::<f32>::from_array(&[1,1], &[1.0]);
	let n2 = Tensor::<f32>::from_array(&[1,1], &[1.0]);

	nn.create_neuron("n1",true);
	nn.create_neuron("n2",true);
	nn.create_neuron("n3",false);

	nn.create_synapse(Vec::new(),
			  "add1", vec!["n1", "n2"], vec!["n3"],
			  | inputs | -> Vec<Tensor<f32>> {
			      vec![inputs[0] + inputs[1]]
			  },
			  | _inputs, grad | -> Vec<Tensor<f32>> {
			      vec![grad.clone(),grad.clone()]
			  });

	nn.forward_prop(vec![n1,n2]);

	println!("n3 {}", nn.ref_neuron("n3").borrow());
    }

    #[test]
    fn nn_test_single_link() {
	let mut nn = NeuralNetwork::<f32>::new();
	let n1 = Tensor::<f32>::from_array(&[1,1], &[0.5]);

	nn.create_neuron("n1",true);

	nn.create_neuron("n2",false);
	nn.create_neuron("n3",false);
	nn.create_neuron("n4",false);

	nn.add_synapse(vec!(),          "square1", vec!["n1"], vec!["n2"], Synapse::<f32>::square());
	nn.add_synapse(vec!["square1"], "exp1",    vec!["n2"], vec!["n3"], Synapse::<f32>::exp());
	nn.add_synapse(vec!["exp1"],    "square2", vec!["n3"], vec!["n4"], Synapse::<f32>::square());

	nn.forward_prop(vec![n1]);
	println!("n4 {}", nn.ref_neuron("n4").borrow());
	nn.backward_prop(true);
	println!("n1 {}", nn.ref_neuron("n1").borrow());
    }

    #[test]
    fn nn_test_add_link(){
	let mut nn = NeuralNetwork::<f32>::new();
	let n1 = Tensor::<f32>::from_array(&[1,1], &[3.0]);

	nn.create_neuron("n1",true);
	nn.create_neuron("n2",false);

	nn.add_synapse(vec!(), "add1", vec!["n1","n1"], vec!["n2"], Synapse::<f32>::add());

	nn.forward_prop(vec![n1]);
	println!("add_link n2 {}", nn.ref_neuron("n2").borrow());
	nn.backward_prop(true);
	println!("add_link n1 {}", nn.ref_neuron("n1").borrow());

	let n1 = Tensor::<f32>::from_array(&[1,1], &[3.0]);
	println!("{}", n1.neg());
    }

    #[test]
    fn nn_test_square(){
	let mut nn = NeuralNetwork::<f32>::new();
	let x = Tensor::<f32>::from_array(&[1,1], &[2.0]);
	let y = Tensor::<f32>::from_array(&[1,1], &[3.0]);

	nn.create_neuron("x",true);
	nn.create_neuron("y",true);

	nn.add_synapse(vec!(), "x^2", vec!["x"],  vec!["x^2"], Synapse::<f32>::square());
	nn.add_synapse(vec!(), "y^2", vec!["y"],  vec!["y^2"], Synapse::<f32>::square());

	nn.add_synapse(vec!["x^2","y^2"], "z", vec!["x^2","y^2"], vec!["z"], Synapse::<f32>::add());

	nn.forward_prop(vec![x,y]);
	nn.backward_prop(false);
	println!("test_sqaure z {}", nn.ref_neuron("z").borrow());
	println!("test_sqaure x {}", nn.ref_neuron("x").borrow());
	println!("test_sqaure y {}", nn.ref_neuron("y").borrow());

    }

    #[test]
    fn nn_test_add_square_link(){
	let mut nn = NeuralNetwork::<f32>::new();
	let x = Tensor::<f32>::from_array(&[1,1], &[2.0]);

	nn.create_neuron("x",true);
	nn.create_neuron("a",false);
	nn.create_neuron("b",false);
	nn.create_neuron("c",false);
	nn.create_neuron("y",false);

	nn.add_synapse(vec!(),                       "square_a", vec!["x"],     vec!["a"], Synapse::<f32>::square());
	nn.add_synapse(vec!["square_a"],             "square_b", vec!["a"],     vec!["b"], Synapse::<f32>::square());
	nn.add_synapse(vec!["square_a"],             "square_c", vec!["a"],     vec!["c"], Synapse::<f32>::square());
	nn.add_synapse(vec!["square_b", "square_c"], "add",      vec!["b","c"], vec!["y"], Synapse::<f32>::add());

	nn.forward_prop(vec![x]);
	nn.backward_prop(true);
	println!("add_square_link y {}", nn.ref_neuron("y").borrow());
	println!("add_square_link x {}", nn.ref_neuron("x").borrow());
	nn.clear_grad();

	let x = Tensor::<f32>::from_array(&[1,1], &[2.0]);
	nn.forward_prop(vec![x]);
	nn.backward_prop(false);
	println!("add_square_link y {}", nn.ref_neuron("y").borrow());
	println!("add_square_link x {}", nn.ref_neuron("x").borrow());
    }

    #[test]
    fn nn_test_sphare(){
	let mut nn = NeuralNetwork::<f32>::new();
	let x = Tensor::<f32>::from_array(&[1,1], &[1.0]);
	let y = Tensor::<f32>::from_array(&[1,1], &[1.0]);
	nn.create_neuron("x",true);
	nn.create_neuron("y",true);
	nn.create_neuron("sqaure_x",true);
	nn.create_neuron("sqaure_y",true);
	nn.create_neuron("z",false);

	nn.add_synapse(vec!(), "square_x", vec!["x"],  vec!["sqaure_x"], Synapse::<f32>::square());
	nn.add_synapse(vec!(), "square_y", vec!["y"],  vec!["sqaure_y"], Synapse::<f32>::square());
	nn.add_synapse(vec!["square_x", "square_y"], "add_x_y", vec!["sqaure_x","sqaure_y"], vec!["z"], Synapse::<f32>::add());

	nn.forward_prop(vec![x,y]);
	nn.backward_prop(false);
	//println!("test_sphare x {:?}", nn.ref_neuron("y"));
	//println!("test_sphare y {:?}", nn.ref_neuron("x"));

	assert_eq!(nn.ref_neuron("x").borrow().grad(), Some(Tensor::<f32>::from_array(&[1,1],&[2.0])));
	assert_eq!(nn.ref_neuron("y").borrow().grad(), Some(Tensor::<f32>::from_array(&[1,1],&[2.0])));
	assert_eq!(nn.ref_neuron("z").borrow().element(vec![0,0]), 2.0);
    }

    #[test]
    fn nn_test_matyas(){
	let mut nn = NeuralNetwork::<f32>::new();
	let x = Tensor::<f32>::from_array(&[1,1], &[1.0]);
	let y = Tensor::<f32>::from_array(&[1,1], &[1.0]);
	let c1 = Neuron::<f32>::create("c1", Tensor::<f32>::from_array(&[1,1], &[0.26]));
	let c2 = Neuron::<f32>::create("c2", Tensor::<f32>::from_array(&[1,1], &[0.48]));

	nn.create_neuron("x",true);
	nn.create_neuron("y",true);
	nn.add_neuron(c1);
	nn.add_neuron(c2);

	nn.create_neuron("square_x",      false);
	nn.create_neuron("square_y",      false);
	nn.create_neuron("mul_x_y",       false);
	nn.create_neuron("add_square",    false);
	nn.create_neuron("c1_add_square", false);
	nn.create_neuron("c2_mul_x_y",    false);
	nn.create_neuron("z",             false);

	nn.add_synapse(vec!(), "square_x", vec!["x"],     vec!["square_x"], Synapse::<f32>::square());
	nn.add_synapse(vec!(), "square_y", vec!["y"],     vec!["square_y"], Synapse::<f32>::square());

	nn.add_synapse(vec!(), "mul_x_y",  vec!["x","y"], vec!["mul_x_y"], Synapse::<f32>::mul());

	nn.add_synapse(vec!["square_x","square_y"],
		       "add_square",
		       vec!["square_x","square_y"],
		       vec!["add_square"], Synapse::<f32>::add());

	nn.add_synapse(vec!["add_square"],
		       "c1_add_square",
		       vec!["c1","add_square"],
		       vec!["c1_add_square"],
		       Synapse::<f32>::mul());
	nn.add_synapse(vec!["mul_x_y"],
		       "c2_mul_x_y",
		       vec!["c2","mul_x_y"],
		       vec!["c2_mul_x_y"],
		       Synapse::<f32>::mul());
	nn.add_synapse(vec!["c1_add_square", "c2_mul_x_y"],
		       "z",
		       vec!["c1_add_square", "c2_mul_x_y"],
		       vec!["z"],
		       Synapse::<f32>::sub());
	nn.forward_prop(vec![x,y]);
	nn.backward_prop(false);

	println!("test_matyas x {}", nn.ref_neuron("x").borrow());
	println!("test_matyas y {}", nn.ref_neuron("y").borrow());
	//println!("test_matyas c1_add_square {:?}", nn.ref_neuron("c1_add_square"));
	//println!("test_matyas c2_mul_x_y {:?}", nn.ref_neuron("c2_mul_x_y"));
	//println!("test_matyas z {}", nn.ref_synapse_node("z").borrow());
	println!("test_matyas z {}", nn.ref_neuron("z").borrow());

	assert_eq!(nn.ref_neuron("z").borrow().element(vec![0,0]), 0.52-0.48);

	if let Some(g) = nn.ref_neuron("x").borrow().grad() {
	    assert_eq!(g[vec![0,0]], 0.52-0.48);
	}
	else {
	    assert!(false)
	}

	if let Some(g) = nn.ref_neuron("y").borrow().grad() {
	    assert_eq!(g[vec![0,0]], 0.52-0.48);
	}
	else {
	    assert!(false)
	}

	assert_eq!(nn.ref_neuron("z").borrow().element(vec![0,0]), 0.52-0.48);
	nn.gen_dot_graph("matyas_graph.dot");
    }


    #[test]
    fn nn_test_backprop_1(){
	let mut nn = NeuralNetwork::<f64>::new();

	nn.create_neuron("x",true);
	nn.create_neuron("y",true);

	nn.add_neuron(Neuron::<f64>::create("2",  Tensor::<f64>::from_array(&[1,1], &[2.0])));
	nn.add_neuron(Neuron::<f64>::create("3",  Tensor::<f64>::from_array(&[1,1], &[3.0])));

	nn.add_synapse(vec!(),            "2*x",         vec!["2","x"],      vec!["2*x"],     Synapse::<f64>::mul());
	nn.add_synapse(vec!(),            "3*y",         vec!["3","y"],      vec!["3*y"],     Synapse::<f64>::mul());
	nn.add_synapse(vec!["2*x","3*y"], "z",           vec!["2*x","3*y"],  vec!["z"],       Synapse::<f64>::sub());

	let x = Tensor::<f64>::from_array(&[1,1], &[1.0]);
	let y = Tensor::<f64>::from_array(&[1,1], &[1.0]);
	nn.forward_prop(vec![x,y]);
	nn.backward_prop(false);

	println!("{}", nn.ref_neuron("x").borrow());
	println!("{}", nn.ref_neuron("y").borrow());
	println!("{}", nn.ref_neuron("z").borrow());

	fn z(x:f64, y:f64) -> f64 {
	    2.0*x - 3.0*y
	}
	println!("{}",z(1.0,1.0));
	println!("{}",(z(1.0+1.0e-4,1.0) - z(1.0-1.0e-4,1.0))/(2.0e-4));
	println!("{}",(z(1.0,1.0+1.0e-4) - z(1.0,1.0-1.0e-4))/(2.0e-4));
    }

    #[test]
    fn nn_test_backprop_2(){
	let mut nn = NeuralNetwork::<f64>::new();

	nn.create_neuron("x",true);
	nn.create_neuron("y",true);

	nn.add_neuron(Neuron::<f64>::create("2",  Tensor::<f64>::from_array(&[1,1], &[2.0])));
	nn.add_neuron(Neuron::<f64>::create("3",  Tensor::<f64>::from_array(&[1,1], &[3.0])));
	nn.add_neuron(Neuron::<f64>::create("16", Tensor::<f64>::from_array(&[1,1], &[16.0])));

	nn.add_synapse(vec!(),                "2*x",         vec!["2","x"],         vec!["2*x"],     Synapse::<f64>::mul());
	nn.add_synapse(vec!(),                "3*y",         vec!["3","y"],         vec!["3*y"],     Synapse::<f64>::mul());
	nn.add_synapse(vec!["2*x","3*y"],     "2*x-3*y",     vec!["2*x","3*y"],     vec!["2*x-3*y"], Synapse::<f64>::sub());
	nn.add_synapse(vec!(),                "x*y",         vec!["x","y"],         vec!["x*y"],     Synapse::<f64>::mul());
	nn.add_synapse(vec!["2*x-3*y","x*y"], "2*x-3*y-x*y", vec!["2*x-3*y","x*y"], vec!["z"],       Synapse::<f64>::sub());

	let x = Tensor::<f64>::from_array(&[1,1], &[1.0]);
	let y = Tensor::<f64>::from_array(&[1,1], &[1.0]);
	nn.forward_prop(vec![x,y]);
	nn.backward_prop(false);

	println!("{}", nn.ref_neuron("x").borrow());
	println!("{}", nn.ref_neuron("y").borrow());
	println!("{}", nn.ref_neuron("z").borrow());

	fn z(x:f64, y:f64) -> f64 {
	    2.0*x - 3.0*y - x*y
	}
	println!("z: {}",z(1.0,1.0));
	println!("dx {}",(z(1.0+1.0e-4,1.0) - z(1.0-1.0e-4,1.0))/(2.0e-4));
	println!("dy {}",(z(1.0,1.0+1.0e-4) - z(1.0,1.0-1.0e-4))/(2.0e-4));

    }
/*
    #[test]
    fn nn_test_backprop_3(){
	let mut nn = NeuralNetwork::<f64>::new();

	nn.create_neuron("x",true);
	nn.create_neuron("y",true);

	nn.add_neuron(Neuron::<f64>::create("1",  Tensor::<f64>::from_array(&[1,1], &[1.0])));
	nn.add_neuron(Neuron::<f64>::create("2",  Tensor::<f64>::from_array(&[1,1], &[2.0])));
	nn.add_neuron(Neuron::<f64>::create("3",  Tensor::<f64>::from_array(&[1,1], &[3.0])));
	nn.add_neuron(Neuron::<f64>::create("16", Tensor::<f64>::from_array(&[1,1], &[16.0])));

	nn.add_synapse(vec!(),                   "2*x",            vec!["2","x"],            vec!["2*x"],            Synapse::<f64>::mul());
	nn.add_synapse(vec!(),                   "3*y",            vec!["3","y"],            vec!["3*y"],            Synapse::<f64>::mul());
	nn.add_synapse(vec!["2*x","3*y"],        "2*x-3*y",        vec!["2*x","3*y"],        vec!["2*x-3*y"],        Synapse::<f64>::sub());
	nn.add_synapse(vec!(),                   "x*y",            vec!["x","y"],            vec!["x*y"],            Synapse::<f64>::mul());
	nn.add_synapse(vec!["x*y"],              "16*x*y",         vec!["16","x*y"],         vec!["16*x*y"],         Synapse::<f64>::mul());
	nn.add_synapse(vec!["2*x-3*y","16*x*y"], "2*x-3*y-16*x*y", vec!["2*x-3*y","16*x*y"], vec!["2*x-3*y-16*x*y"], Synapse::<f64>::sub());
	nn.add_synapse(vec!["x*y"],              "1+x*y",          vec!["1","x*y"],          vec!["1+x*y"],          Synapse::<f64>::add());
	nn.add_synapse(vec!["2*x-3*y-16*x*y","1+x*y"],
		       "(2*x-3*y-16*x*y)*(1+x*y)", vec!["2*x-3*y-16*x*y","1+x*y"], vec!["z"], Synapse::<f64>::mul());


	let x = Tensor::<f64>::from_array(&[1,1], &[1.0]);
	let y = Tensor::<f64>::from_array(&[1,1], &[1.0]);
	nn.forward_prop(vec![x,y]);
	nn.backward_prop(false);

	println!("{:?}", nn.ref_neuron("x"));
	println!("{:?}", nn.ref_neuron("y"));
	println!("{:?}", nn.ref_neuron("z"));

	fn z(x:f64, y:f64) -> f64 {
	    (2.0*x - 3.0*y - 16.0*x*y)*(1.0+x*y)
	}
	println!("z: {}",z(1.0,1.0));
	println!("dx {}",(z(1.0+1.0e-4,1.0) - z(1.0-1.0e-4,1.0))/(2.0e-4));
	println!("dy {}",(z(1.0,1.0+1.0e-4) - z(1.0,1.0-1.0e-4))/(2.0e-4));

    }
*/
    #[test]
    fn nn_test_backprop_4(){
	let mut nn = NeuralNetwork::<f64>::new();

	nn.create_neuron("x",true);
	nn.create_neuron("y",true);

	nn.add_neuron(Neuron::<f64>::create("1",  Tensor::<f64>::from_array(&[1,1], &[1.0])));
	nn.add_neuron(Neuron::<f64>::create("2",  Tensor::<f64>::from_array(&[1,1], &[2.0])));
	nn.add_neuron(Neuron::<f64>::create("3",  Tensor::<f64>::from_array(&[1,1], &[3.0])));
	nn.add_neuron(Neuron::<f64>::create("16", Tensor::<f64>::from_array(&[1,1], &[16.0])));

	nn.add_synapse(vec!(),                   "2*x",            vec!["2","x"],            vec!["2*x"],      Synapse::<f64>::mul());
	nn.add_synapse(vec!(),                   "3*y",            vec!["3","y"],            vec!["3*y"],      Synapse::<f64>::mul());
	nn.add_synapse(vec!["2*x","3*y"],        "2*x-3*y",        vec!["2*x","3*y"],        vec!["2*x-3*y"],  Synapse::<f64>::sub());
	nn.add_synapse(vec!["2*x-3*y"],          "(2*x-3*y)^2",    vec!["2*x-3*y"],          vec!["z"],        Synapse::<f64>::square());

	let x = Tensor::<f64>::from_array(&[1,1], &[1.0]);
	let y = Tensor::<f64>::from_array(&[1,1], &[1.0]);
	nn.forward_prop(vec![x,y]);
	nn.backward_prop(false);

	println!("backprop_4 {}", nn.ref_neuron("x").borrow());
	println!("backprop_4 {}", nn.ref_neuron("y").borrow());
	println!("backprop_4 {}", nn.ref_neuron("z").borrow());

	fn z(x:f64, y:f64) -> f64 {
	    (2.0*x - 3.0*y).powf(2.0)
	}
	println!("backprop_4 z: {}",z(1.0,1.0));
	println!("backprop_4 dx {}",(z(1.0+1.0e-4,1.0) - z(1.0-1.0e-4,1.0))/(2.0e-4));
	println!("backprop_4 dy {}",(z(1.0,1.0+1.0e-4) - z(1.0,1.0-1.0e-4))/(2.0e-4));

    }

    #[test]
    fn nn_test_goldstein(){
	let mut nn = NeuralNetwork::<f64>::new();

	nn.create_neuron("x",true);
	nn.create_neuron("y",true);

	nn.add_neuron(Neuron::<f64>::create("1",  Tensor::<f64>::from_array(&[1,1], &[1.0])));
	nn.add_neuron(Neuron::<f64>::create("2",  Tensor::<f64>::from_array(&[1,1], &[2.0])));
	nn.add_neuron(Neuron::<f64>::create("3",  Tensor::<f64>::from_array(&[1,1], &[3.0])));
	nn.add_neuron(Neuron::<f64>::create("6",  Tensor::<f64>::from_array(&[1,1], &[6.0])));
	nn.add_neuron(Neuron::<f64>::create("12", Tensor::<f64>::from_array(&[1,1], &[12.0])));
	nn.add_neuron(Neuron::<f64>::create("14", Tensor::<f64>::from_array(&[1,1], &[14.0])));
	nn.add_neuron(Neuron::<f64>::create("18", Tensor::<f64>::from_array(&[1,1], &[18.0])));
	nn.add_neuron(Neuron::<f64>::create("19", Tensor::<f64>::from_array(&[1,1], &[19.0])));
	nn.add_neuron(Neuron::<f64>::create("27", Tensor::<f64>::from_array(&[1,1], &[27.0])));
	nn.add_neuron(Neuron::<f64>::create("30", Tensor::<f64>::from_array(&[1,1], &[30.0])));
	nn.add_neuron(Neuron::<f64>::create("32", Tensor::<f64>::from_array(&[1,1], &[32.0])));
	nn.add_neuron(Neuron::<f64>::create("36", Tensor::<f64>::from_array(&[1,1], &[36.0])));
	nn.add_neuron(Neuron::<f64>::create("48", Tensor::<f64>::from_array(&[1,1], &[48.0])));

	nn.add_synapse(vec!(),          "x+y",       vec!["x","y"],                vec!["x+y"],       Synapse::<f64>::add());
	nn.add_synapse(vec!["x+y"],     "x+y+1",     vec!["x+y","1"],              vec!["x+y+1"],     Synapse::<f64>::add());
	nn.add_synapse(vec!["x+y+1"],   "(x+y+1)^2", vec!["x+y+1"],                vec!["(x+y+1)^2"], Synapse::<f64>::square());
	nn.add_synapse(vec!(),          "14*x",      vec!["14","x"],               vec!["14*x"],      Synapse::<f64>::mul());
	nn.add_synapse(vec!["14*x"],    "19-14*x",   vec!["19","14*x"],            vec!["19-14*x"],   Synapse::<f64>::sub());
	nn.add_synapse(vec!(),          "x^2",       vec!["x"],                    vec!["x^2"],       Synapse::<f64>::square());
	nn.add_synapse(vec!["x^2"],     "3*(x^2)",   vec!["3","x^2"],              vec!["3*(x^2)"],   Synapse::<f64>::mul());
	nn.add_synapse(vec!(),          "14*y",      vec!["14","y"],               vec!["14*y"],      Synapse::<f64>::mul());
	nn.add_synapse(vec!(),          "x*y",       vec!["x","y"],                vec!["x*y"],       Synapse::<f64>::mul());
	nn.add_synapse(vec!["x*y"],     "6*x*y",     vec!["6","x*y"],              vec!["6*x*y"],     Synapse::<f64>::mul());
	nn.add_synapse(vec!(),          "y^2",       vec!["y"],                    vec!["y^2"],       Synapse::<f64>::square());
	nn.add_synapse(vec!["y^2"],     "3*(y^2)",   vec!["3","y^2"],              vec!["3*(y^2)"],   Synapse::<f64>::mul());

	nn.add_synapse(vec!["19-14*x", "3*(x^2)"],
		       "19-14*x+3*(x^2)",
		       vec!["19-14*x", "3*(x^2)"],
		       vec!["19-14*x+3*(x^2)"],
		       Synapse::<f64>::add());
	nn.add_synapse(vec!["19-14*x+3*(x^2)","14*y"],
		       "19-14*x+3*(x^2)-14*y",
		       vec!["19-14*x+3*(x^2)","14*y"],
		       vec!["19-14*x+3*(x^2)-14*y"],
		       Synapse::<f64>::sub());

	nn.add_synapse(vec!["19-14*x+3*(x^2)-14*y","6*x*y"],
		       "19-14*x+3*(x^2)-14*y+6*x*y",
		       vec!["19-14*x+3*(x^2)-14*y","6*x*y"],
		       vec!["19-14*x+3*(x^2)-14*y+6*x*y"],
		       Synapse::<f64>::add());

	nn.add_synapse(vec!["19-14*x+3*(x^2)-14*y+6*x*y","3*(y^2)"],
		       "19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2)",
		       vec!["19-14*x+3*(x^2)-14*y+6*x*y","3*(y^2)"],
		       vec!["19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2)"],
		       Synapse::<f64>::add());

	nn.add_synapse(vec!["(x+y+1)^2","19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2)"],
		       "((x+y+1)^2)*(19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2))",
		       vec!["(x+y+1)^2", "19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2)"],
		       vec!["((x+y+1)^2)*(19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2))"],
		       Synapse::<f64>::mul());
	nn.add_synapse(vec!["((x+y+1)^2)*(19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2))"],
		       "1+((x+y+1)^2)*(19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2))",
		       vec!["1","((x+y+1)^2)*(19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2))"],
		       vec!["1+((x+y+1)^2)*(19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2))"],
		       Synapse::<f64>::add());

	nn.add_synapse(vec!(),            "2*x",         vec!["2","x"],        vec!["2*x"],         Synapse::<f64>::mul());
	nn.add_synapse(vec!(),            "3*y",         vec!["3","y"],        vec!["3*y"],         Synapse::<f64>::mul());
	nn.add_synapse(vec!["2*x","3*y"], "2*x-3*y",     vec!["2*x","3*y"],    vec!["2*x-3*y"],     Synapse::<f64>::sub());
	nn.add_synapse(vec!["2*x-3*y"],   "(2*x-3*y)^2", vec!["2*x-3*y"],      vec!["(2*x-3*y)^2"], Synapse::<f64>::square());
	nn.add_synapse(vec!(),            "32*x",        vec!["32","x"],       vec!["32*x"],        Synapse::<f64>::mul());
	nn.add_synapse(vec!["32*x"],      "18-32*x",     vec!["18","32*x"],    vec!["18-32*x"],     Synapse::<f64>::sub());
	nn.add_synapse(vec!["x^2"],       "12*(x^2)",    vec!["12","x^2"],     vec!["12*(x^2)"],    Synapse::<f64>::mul());
	nn.add_synapse(vec!(),            "48*y",        vec!["48","y"],       vec!["48*y"],        Synapse::<f64>::mul());
	nn.add_synapse(vec!["x*y"],       "36*x*y",      vec!["36","x*y"],     vec!["36*x*y"],      Synapse::<f64>::mul());
	nn.add_synapse(vec!["y^2"],       "27*(y^2)",    vec!["27","y^2"],     vec!["27*(y^2)"],    Synapse::<f64>::mul());

	nn.add_synapse(vec!["18-32*x","12*(x^2)"],
		       "18-32*x+12*(x^2)",
		       vec!["18-32*x","12*(x^2)"],
		       vec!["18-32*x+12*(x^2)"],
		       Synapse::<f64>::add());

	nn.add_synapse(vec!["18-32*x+12*(x^2)","48*y"],
		       "18-32*x+12*(x^2)+48*y",
		       vec!["18-32*x+12*(x^2)","48*y"],
		       vec!["18-32*x+12*(x^2)+48*y"],
		       Synapse::<f64>::add());

	nn.add_synapse(vec!["18-32*x+12*(x^2)+48*y","36*x*y"],
		       "18-32*x+12*(x^2)+48*y-36*x*y",
		       vec!["18-32*x+12*(x^2)+48*y","36*x*y"],
		       vec!["18-32*x+12*(x^2)+48*y-36*x*y"],
		       Synapse::<f64>::sub());

	nn.add_synapse(vec!["18-32*x+12*(x^2)+48*y-36*x*y","27*(y^2)"],
		       "18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2)",
		       vec!["18-32*x+12*(x^2)+48*y-36*x*y","27*(y^2)"],
		       vec!["18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2)"],
		       Synapse::<f64>::add());

	nn.add_synapse( vec!["(2*x-3*y)^2","18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2)"],
			"(2*x-3*y)^2*(18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2))",
			vec!["(2*x-3*y)^2","18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2)"],
			vec!["(2*x-3*y)^2*(18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2))"],
			Synapse::<f64>::mul());

	nn.add_synapse( vec!["(2*x-3*y)^2*(18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2))"],
			"30+(2*x-3*y)^2*(18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2))",
			vec!["30","(2*x-3*y)^2*(18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2))"],
			vec!["30+(2*x-3*y)^2*(18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2))"],
			Synapse::<f64>::add());

	nn.add_synapse(vec!["1+((x+y+1)^2)*(19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2))",
			    "30+(2*x-3*y)^2*(18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2))"],
		       "z",
		       vec!["1+((x+y+1)^2)*(19-14*x+3*(x^2)-14*y+6*x*y+3*(y^2))",
			    "30+(2*x-3*y)^2*(18-32*x+12*(x^2)+48*y-36*x*y+27*(y^2))"],
		       vec!["z"],
		       Synapse::<f64>::mul());

	let x = Tensor::<f64>::from_array(&[1,1], &[1.0]);
	let y = Tensor::<f64>::from_array(&[1,1], &[1.0]);
	nn.forward_prop(vec![x,y]);
	nn.backward_prop(false);

	//println!("backprop_5 {:?}", nn.ref_neuron("x"));
	//println!("backprop_5 {:?}", nn.ref_neuron("y"));
	//println!("backprop_5 {:?}", nn.ref_neuron("z"));

	fn z(x:f64, y:f64) -> f64 {
	    (1.0+ ((x+y+1.0).powf(2.0))*(19.0 - 14.0*x + 3.0*x.powf(2.0) - 14.0*y + 6.0*x*y + 3.0*y.powf(2.0))) *
		(30.0+(2.0*x-3.0*y).powf(2.0)*(18.0 - 32.0*x + 12.0*x.powf(2.0) + 48.0*y - 36.0*x*y + 27.0*y.powf(2.0)))
	}

	//println!("backprop_5 z: {}",z(1.0,1.0));
	//println!("backprop_5 dx {}",(z(1.0+1.0e-4,1.0) - z(1.0-1.0e-4,1.0))/(2.0e-4));
	//println!("backprop_5 dy {}",(z(1.0,1.0+1.0e-4) - z(1.0,1.0-1.0e-4))/(2.0e-4));

	assert_eq!(z(1.0,1.0), nn.ref_neuron("z").borrow().element(vec![0,0]));

	if let Some(g) = nn.ref_neuron("x").borrow().grad() {
	    let diff = ((z(1.0+1.0e-4,1.0) - z(1.0-1.0e-4,1.0))/(2.0e-4) - g[vec![0,0]]).abs();
	    assert!(diff < 1e-3);
	}
	else {
	    assert!(false);
	};

	if let Some(g) = nn.ref_neuron("y").borrow().grad() {
	    let diff = ((z(1.0,1.0+1.0e-4) - z(1.0,1.0-1.0e-4))/(2.0e-4) - g[vec![0,0]]).abs();
	    assert!(diff < 1e-3);
	}
	else {
	    assert!(false);
	};

	nn.gen_dot_graph("gs_graph.dot");
    }

    #[test]
    fn nn_test_pow(){
	let mut nn = NeuralNetwork::<f64>::new();

	nn.create_neuron("x",true);
	nn.create_neuron("y",true);

	nn.add_neuron(Neuron::<f64>::create("2",  Tensor::<f64>::from_array(&[1,1], &[2.0])));
	nn.add_neuron(Neuron::<f64>::create("3",  Tensor::<f64>::from_array(&[1,1], &[3.0])));
	nn.add_neuron(Neuron::<f64>::create("6",  Tensor::<f64>::from_array(&[1,1], &[6.0])));
	nn.add_neuron(Neuron::<f64>::create("12", Tensor::<f64>::from_array(&[1,1], &[12.0])));

	nn.add_synapse(vec!(), "x^2", vec!["x","2"],  vec!["x^2"], Synapse::<f64>::pow());
	nn.add_synapse(vec!(), "y^3", vec!["y","3"],  vec!["y^3"], Synapse::<f64>::pow());
	nn.add_synapse(vec!["x^2"], "12*(x^2)", vec!["12","x^2"], vec!["12*(x^2)"], Synapse::<f64>::mul());
	nn.add_synapse(vec!["y^3"], "6*(y^3)",  vec!["6", "y^3"], vec!["6*(y^3)"],  Synapse::<f64>::mul());
	nn.add_synapse(vec!["12*(x^2)","6*(y^3)"], "z", vec!["12*(x^2)", "6*(y^3)"], vec!["z"],  Synapse::<f64>::add());

	let x = Tensor::<f64>::from_array(&[1,1], &[2.0]);
	let y = Tensor::<f64>::from_array(&[1,1], &[2.0]);

	nn.forward_prop(vec![x,y]);
	nn.backward_prop(false);

	println!("nn_test_pow {}", nn.ref_neuron("x").borrow());
	println!("nn_test_pow {}", nn.ref_neuron("y").borrow());
	println!("nn_test_pow {}", nn.ref_neuron("z").borrow());

	fn z(x:f64, y:f64) -> f64 {
	    12.0*x.powf(2.0) + 6.0*y.powf(3.0)
	}

	assert_eq!(z(2.0,2.0), nn.ref_neuron("z").borrow().element(vec![0,0]));

	if let Some(g) = nn.ref_neuron("x").borrow().grad() {
	    let diff = ((z(2.0+1.0e-4,2.0) - z(2.0-1.0e-4,2.0))/(2.0e-4) - g[vec![0,0]]).abs();
	    assert!(diff < 1e-3);
	}
	else {
	    assert!(false);
	};

	if let Some(g) = nn.ref_neuron("y").borrow().grad() {
	    let diff = ((z(2.0,2.0+1.0e-4) - z(2.0,2.0-1.0e-4))/(2.0e-4) - g[vec![0,0]]).abs();
	    assert!(diff < 1e-3);
	}
	else {
	    assert!(false);
	};

    }

    #[test]
    fn nn_test_taylor_expand(){
	let mut nn = NeuralNetwork::<f64>::new();
	let mut counter = 1;
	let mut result_label = "".to_string();
	nn.create_neuron("x",true).borrow_mut().assign(Tensor::<f64>::from_array(&[1,1], &[std::f64::consts::PI/4.0]));
	nn.add_neuron(Neuron::<f64>::create("0",  Tensor::<f64>::from_array(&[1,1], &[0.0])));

	loop {
	    let label = "term_".to_string()+&counter.to_string();
	    let sum_label = "sum_".to_string()+&counter.to_string();
	    let power_int = (counter-1)*2 + 1;
	    let power = power_int as f64;
	    if counter == 1 {
		let const_label = power_int.to_string();
		nn.add_neuron(Neuron::<f64>::create(&const_label,  Tensor::<f64>::from_array(&[1,1], &[power])));
		nn.add_synapse(vec!(),       &label,     vec!["x",&const_label], vec![&label],     Synapse::<f64>::pow());
		nn.add_synapse(vec![&label], &sum_label, vec!["0", &label],      vec![&sum_label], Synapse::<f64>::add());
	    }
	    else {
		let const_label = power_int.to_string();
		let factorial:f64 = (1..(power_int+1)).fold(1.0, |p, n| p * (n as f64));
		let c:f64 = (-1.0 as f64).powi(counter-1)/factorial;
		let coff_label = "coff_".to_string()+&counter.to_string();
		let power_label = "power_".to_string()+&power_int.to_string();
		let prev_sum_label = "sum_".to_string()+&(counter-1).to_string();
		// c = (-1)^(counter-1)/power_int!
		nn.add_neuron(Neuron::<f64>::create(&coff_label,  Tensor::<f64>::from_array(&[1,1], &[c])));
		nn.add_neuron(Neuron::<f64>::create(&const_label, Tensor::<f64>::from_array(&[1,1], &[power])));
		//println!("fact {}! {}\n", power_int, factorial);
		nn.add_synapse(vec!(), &power_label, vec!["x",&const_label],  vec![&power_label], Synapse::<f64>::pow());
		nn.add_synapse(vec![&power_label], &label, vec![&coff_label, &power_label], vec![&label], Synapse::<f64>::mul());
		nn.add_synapse(vec![&prev_sum_label, &label], &sum_label, vec![&prev_sum_label, &label],  vec![&sum_label], Synapse::<f64>::add());
		result_label = sum_label;
	    };
	    //println!("{} {}", label, nn.ref_neuron(&label).borrow());
	    if nn.ref_neuron(&label).borrow().element(vec![0,0]).abs() < 1.0e-4 {
		/*
		println!("{} {} {}",
			 nn.ref_neuron(&label).borrow().element(vec![0,0]),
			 nn.ref_neuron(&label).borrow().element(vec![0,0]) < 1.0e-4,
			 1.0e-4);
		*/
		break;
	    }
	    counter += 1;
	}


	println!("taylor: {} {}",nn.ref_neuron(&result_label).borrow().element(vec![0,0]),(std::f64::consts::PI/4.0).sin());
	let diff = (nn.ref_neuron(&result_label).borrow().element(vec![0,0])-(std::f64::consts::PI/4.0).sin()).abs();
	assert!(diff < 1.0e-5);
	nn.backward_prop(false);

	if let Some(g) = nn.ref_neuron("x").borrow().grad() {
	    let diff = (g[vec![0,0]]-(std::f64::consts::PI/4.0).cos()).abs();
	    println!("taylor: {} {}",g[vec![0,0]],(std::f64::consts::PI/4.0).cos());
	    assert!(diff < 1.0e-5);
	}
	else {
	    assert!(false);
	}
	nn.gen_dot_graph("taylor_graph.dot");
    }
}

