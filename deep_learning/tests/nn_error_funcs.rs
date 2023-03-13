/* -*- tab-width:4 -*- */

use std::fmt::Display;
use std::rc::Rc;
use std::error::Error;

use rand::SeedableRng;
use rand_distr::{Normal, Uniform, Distribution};
use rand_xorshift::XorShiftRng;

use deep_learning::neural_network::NeuralNetwork;
use linear_transform::Tensor;

use plotters::prelude::{Circle, BitMapBackend,ChartBuilder,PathElement};
use plotters::prelude::{SeriesLabelPosition};
use plotters::prelude::full_palette::*;
use plotters::drawing::IntoDrawingArea;
use plotters::style::{IntoFont,Color};

#[derive(Debug)]
enum MyError {
	StringMsg(String)
}

impl Display for MyError {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		use self::MyError::*;
		match self {
			StringMsg(s) => write!(f, "{}", s)
		}
	}
}

impl Error for MyError {}

fn numerical_grad(f:&(dyn Fn(&Tensor::<f64>) -> Tensor::<f64>),
				  t:&Tensor::<f64>,
				  delta:f64 ) -> Tensor::<f64> {

	let mut pos:usize = 0;
	let mut v:Vec<f64> = Vec::new();

	for x in t.buffer().iter() {
		let mut forward_t  = t.clone();
		let mut backward_t = t.clone();

		forward_t.replace_element_by_index(pos,  x-delta);
		backward_t.replace_element_by_index(pos, x+delta);

		let forward_ft  = f(&forward_t);
		let backward_ft = f(&backward_t);

		let diff = (backward_ft - forward_ft).sum(&[1,1])[vec![0,0]];
		v.push(diff/(2.0*delta));
		pos += 1;
	}

	Tensor::<f64>::from_vector(t.shape().to_vec(), v)
}

#[test]
fn sce_test() -> Result<(),Box<dyn std::error::Error>> {

	fn softmax(t:&Tensor<f64>, axis:usize) -> Tensor<f64> {
		let src_shape = t.shape();
		let max = t.max_in_axis(axis).broadcast(src_shape);
		let y = (t - max).exp();
		let sum_y = y.sum_axis(axis).broadcast(src_shape);
		Tensor::<f64>::hadamard_division(&y, &sum_y)
	}

	fn softmax_cross_entropy(x:&Tensor<f64>, t:&Tensor<f64>) -> Tensor<f64> {
		let orig_shape = x.shape().to_vec();
		let m = x.max_in_axis(1);
		let y = x - m.broadcast(&orig_shape);
		let y = y.exp();
		let s = y.sum_axis(1);
		let s = s.ln();
		let log_z = m + s;
		let log_p = x - log_z.broadcast(&orig_shape);
		let t_line = t.ravel();
		//println!("t_line {t_line}");
		let label = (0..orig_shape[0]).map(|i| i as f64).collect::<Vec<f64>>();
		let label = Tensor::<f64>::from_vector(vec![1,orig_shape[0]], label);
		//println!("label {}",label);
		let log_p_v = label.buffer().iter().zip(t_line.buffer().iter()).map(|(l,t)| {
			log_p[vec![*l as usize ,*t as usize]]
		}).collect::<Vec<f64>>();
		//println!("log_p_v: {:?}", log_p_v);
		let log_p_sum = log_p_v.iter().fold(0.0,|s,&e| { s + e });
		Tensor::<f64>::from_array(&[1,1],&[-log_p_sum/(orig_shape[0] as f64)])
	}

	{
		let x_init = Tensor::<f64>::from_array(&[2,4],
											   &[-1.0, 0.0, 1.0,  2.0,
												 2.0, 0.0, 1.0, -1.0]);
		let t_init = Tensor::<f64>::from_array(&[1,2], &[3.0, 0.0]);
		let y = softmax_cross_entropy(&x_init,&t_init);
		let delta = 1.0e-4;

		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", x_init.clone());
		let t = nn.create_neuron("t", t_init.clone());
		let err = nn.softmax_cross_entropy_error(Rc::clone(&x),Rc::clone(&t));
		err.borrow_mut().rename("err");
		println!("{}",err.borrow());
		let result = err.borrow().ref_signal().buffer().iter().zip(y.buffer().iter())
			.fold(false, |b,(e,y)| { b || !((e-y).abs() < delta) });
		if result {
			return Err(Box::new(MyError::StringMsg("invalid function result".to_string())));
		}
		nn.backward_propagating(0)?;
		let numgrad = numerical_grad(&|x:&Tensor<f64>| { softmax_cross_entropy(x, &t_init) }, &x_init, 1.0e-4);
		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			let result =
				numgrad.buffer().iter().zip(gx.borrow().ref_signal().buffer().iter())
				.fold(false,|b, (x0,x1)| { b || !((x0 - x1).abs() < delta) });
			if result {
				return Err(Box::new(MyError::StringMsg("invalid grad".to_string())));
			}
		}
		else {
			return Err(Box::new(MyError::StringMsg("no grad".to_string())));
		}
	}

	{
		let x_init = Tensor::<f64>::from_array(&[3,4],
											   &[-1.0, 0.0, 1.0,  2.0,
												 2.0, 0.0, 1.0, -1.0,
												 1.0, 2.0, 1.0, -1.0]);
		let t_init = Tensor::<f64>::from_array(&[1,3], &[3.0, 0.0, 1.0]);
		let y = softmax_cross_entropy(&x_init,&t_init);
		let delta = 1.0e-4;

		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", x_init.clone());
		let t = nn.create_neuron("t", t_init.clone());
		let err = nn.softmax_cross_entropy_error(Rc::clone(&x),Rc::clone(&t));
		err.borrow_mut().rename("err");
		println!("{}",err.borrow());
		let result = err.borrow().ref_signal().buffer().iter().zip(y.buffer().iter())
			.fold(false, |b,(e,y)| { b || !((e-y).abs() < delta) });
		if result {
			return Err(Box::new(MyError::StringMsg("invalid function result".to_string())));
		}
		nn.backward_propagating(0)?;
		let numgrad = numerical_grad(&|x:&Tensor<f64>| { softmax_cross_entropy(x, &t_init) }, &x_init, 1.0e-4);
		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			let result =
				numgrad.buffer().iter().zip(gx.borrow().ref_signal().buffer().iter())
				.fold(false,|b, (x0,x1)| { b || !((x0 - x1).abs() < delta) });
			if result {
				return Err(Box::new(MyError::StringMsg("invalid grad".to_string())));
			}
		}
		else {
			return Err(Box::new(MyError::StringMsg("no grad".to_string())));
		}
	}

	{
		let mut nn = NeuralNetwork::<f64>::new();
		let delta = 1.0e-4;
		let uniform_dist = Uniform::from(0..10);
		let t_src = (0..100).map(|_| { uniform_dist.sample(&mut nn.get_rng()) as f64 }).collect::<Vec<f64>>();
		let x = nn.create_normal_distribution_neuron("x", &[100,10], 0.0, 1.0);
		let t = nn.create_constant("t", Tensor::<f64>::from_vector(vec![1,100], t_src));
		let y = softmax_cross_entropy(x.borrow().ref_signal(),t.borrow().ref_signal());
		let err = nn.softmax_cross_entropy_error(Rc::clone(&x),Rc::clone(&t));
		let result = err.borrow().ref_signal().buffer().iter().zip(y.buffer().iter())
			.fold(false, |b,(e,y)| { b || !((e-y).abs() < delta) });
		if result {
			return Err(Box::new(MyError::StringMsg("invalid function result".to_string())));
		}

		nn.backward_propagating(0)?;
		let numgrad = numerical_grad(&|x:&Tensor<f64>| { softmax_cross_entropy(x, t.borrow().ref_signal()) },
									 x.borrow().ref_signal(), 1.0e-4);
		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			let result =
				numgrad.buffer().iter().zip(gx.borrow().ref_signal().buffer().iter())
				.fold(false,|b, (x0,x1)| { b || !((x0 - x1).abs() < delta) });
			if result {
				return Err(Box::new(MyError::StringMsg("invalid grad".to_string())));
			}
		}
		else {
			return Err(Box::new(MyError::StringMsg("no grad".to_string())));
		}
	}

	Ok(())
}
