use std::ops::Deref;
use std::cell::{Ref,RefCell};
use std::rc::Rc;
use linear_transform::tensor::tensor_base::Tensor;

pub struct Synapse<T>
    where T:num::Float + num::pow::Pow<T, Output = T> + Clone + std::fmt::Debug {
    forward : fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
    backward: fn (inputs: &Vec<&Tensor<T>>,grad:&Tensor<T>) -> Vec<Tensor<T>>,
}

impl<T> Synapse<T>
where T:num::Float + num::FromPrimitive + num::pow::Pow<T, Output = T> + Clone + std::fmt::Debug {

    pub fn new(forward : fn (inputs: &Vec<&Tensor<T>>) -> Vec<Tensor<T>>,
	       backward: fn (inputs: &Vec<&Tensor<T>>, grad:&Tensor<T>) -> Vec<Tensor<T>>) -> Synapse<T> {
	Synapse {
	    forward,
	    backward,
	}
    }

    pub fn forward(&mut self,inputs: &Vec<Rc<RefCell<Tensor<T>>>>) -> Vec<Tensor<T>> {
	let holder:Vec<Ref<'_,Tensor<T>>> = inputs.iter().map(|i| { i.borrow() }).collect();
	let inputs_ref:Vec<&Tensor<T>> = holder.iter().map(|h| { h.deref() }).collect();
	((*self).forward)(&inputs_ref)
    }

    pub fn backward(&mut self,
		    inputs: &Vec<Rc<RefCell<Tensor<T>>>>,
		    grad:  Rc<RefCell<Tensor<T>>>) -> Vec<Tensor<T>> {
	let inputs_holder:Vec<Ref<'_,Tensor<T>>> = inputs.iter().map(|i| { i.borrow() }).collect();
	let inputs_ref:Vec<&Tensor<T>> = inputs_holder.iter().map(|h| { h.deref() }).collect();
	((*self).backward)(&inputs_ref, grad.borrow().deref())
    }

    pub fn add() -> Synapse<T> {
	Synapse {
	    forward:  |inputs| { vec![inputs[0] + inputs[1]] },
	    backward: |_inputs, grad| { vec![grad.clone(), grad.clone()] }
	}
    }

    pub fn neg() -> Synapse<T> {
	Synapse {
	    forward:  |inputs| { vec![inputs[0].neg()] },
	    backward: |_inputs, grad| { vec![grad.neg()] }
	}
    }

    pub fn sub() -> Synapse<T> {
	Synapse {
	    forward:  |inputs| { vec![inputs[0] - inputs[1]] },
	    backward: |_inputs, grad| { vec![grad.clone(), grad.neg()] }
	}
    }

    pub fn mul() -> Synapse<T> {
	Synapse {
	    forward: |inputs| {
		vec![Tensor::<T>::mul_rank0(inputs[0], inputs[1])]
	    },
	    backward: |inputs, grad| { vec![ Tensor::<T>::mul_rank0(inputs[1],grad),
					     Tensor::<T>::mul_rank0(inputs[0],grad) ] }
	}
    }

    pub fn div() -> Synapse<T> {
	Synapse {
	    forward: |inputs| {
		vec![Tensor::<T>::div_rank0(inputs[0], inputs[1])]
	    },
	    backward: |inputs, grad| {
		let two = num::FromPrimitive::from_f64(2.0).unwrap();
		let gl = Tensor::<T>::div_rank0(grad,inputs[1]);
		let gr = Tensor::<T>::mul_rank0(grad, &Tensor::<T>::div_rank0(&inputs[0].neg(), &inputs[1].pow_rank0(two)));
		vec![ gl, gr ]
	    }
	}
    }

    pub fn square() -> Synapse<T> {
	Synapse {
	    forward:  |inputs| { inputs.iter().map(|i| i.square()).collect() },
	    backward: |inputs, grad| { inputs.iter().map(|i| i.scale( num::FromPrimitive::from_f64(2.0).unwrap() ).scale(grad[vec![0,0]])).collect() }
	}
    }

    pub fn exp() -> Synapse<T> {
	Synapse {
	    forward:  |inputs| { inputs.iter().map(|i| i.exp()).collect() },
	    backward: |inputs,grad| { inputs.iter().map(|i| i.exp().scale(grad[vec![0,0]])).collect() }
	}
    }

    pub fn pow() -> Synapse<T> {
	Synapse {
	    forward: |inputs| {
		// first index is base
		// second index is exp
		vec![inputs[0].pow_rank0(inputs[1][vec![0,0]])]
	    },
	    backward: |inputs,grad| {
		let dinput_0 = inputs[0].pow_rank0(inputs[1][vec![0,0]]-num::one())
		    .scale(inputs[1][vec![0,0]])
		    .scale(grad[vec![0,0]]);
		let dinput_1 = inputs[0].pow_rank0(inputs[1][vec![0,0]]).scale(inputs[0][vec![0,0]].ln());
		vec![dinput_0,dinput_1]
	    }
	}
    }
}
