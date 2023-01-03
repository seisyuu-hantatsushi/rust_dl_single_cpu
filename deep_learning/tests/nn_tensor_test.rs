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

fn differntial(f: &(dyn Fn(&Tensor::<f64>) -> Tensor::<f64>),
			   x:&Tensor::<f64>,
			   delta:f64 ) -> Tensor::<f64> {
	let x_shape = x.shape();
	let mut diff = Tensor::<f64>::zero(x_shape);

	for i in 0..x.num_of_elements() {
		let mut x_forward  = x.clone();
		let mut x_backward = x.clone();

		let pos = x.index_to_position(i);
		x_forward[pos.clone()]  = x_forward[pos.clone()]  + delta;
		x_backward[pos.clone()] = x_backward[pos.clone()] - delta;

		let y_forward  = f(&x_forward);
		let y_backward = f(&x_backward);

		let dt = (y_forward - y_backward).scale(1.0/(2.0*delta));

		if dt.num_of_elements() == 1 {
			diff[pos.clone()] = dt[vec![0,0]];
		}
		else {
			if x.shape() != dt.shape() {
				let t = dt.sum(x.shape());
				diff[pos.clone()] = t[pos.clone()];
			}
			else {
				diff[pos.clone()] = dt[pos.clone()];
			}
		}
	}
	diff
}

#[test]
fn reshape_test() -> Result<(),Box<dyn std::error::Error>> {
    let mut nn = NeuralNetwork::<f64>::new();
    let xs:Vec<f64> = (1..=12).map(|i| (i as f64)).collect();
    let x  = nn.create_neuron("x", Tensor::<f64>::from_vector(vec![2,6],xs));
    let x0 = nn.reshape(x,vec![3,4]);

    println!("{}", x0.borrow());
    assert_eq!(&vec![3,4], x0.borrow().ref_signal().shape());
    Ok(())
}

#[test]
fn transpose_test() -> Result<(),Box<dyn std::error::Error>> {
    let mut nn = NeuralNetwork::<f64>::new();
    let xs:Vec<f64> = (0..36).map(|i| ((i/6)*10 + (i % 6) + 11) as f64).collect();
	let txs:Vec<f64> = (0..36).map(|i| ((i % 6)*10 + (i/6) + 11) as f64).collect();
    let x  = nn.create_neuron("x", Tensor::<f64>::from_vector(vec![6,6],xs));
	let transpose_x = nn.transpose(Rc::clone(&x));

	let tx = Tensor::<f64>::from_vector(vec![6,6],txs);
	/*
	println!("{}", x.borrow());
    println!("{}", transpose_x.borrow());
	println!("{}", tx);
	 */
	assert_eq!(transpose_x.borrow().ref_signal(), &tx);

    let xs:Vec<f64> = (0..42).map(|i| ((i/7)*10 + (i % 7) + 11) as f64).collect();
	let txs:Vec<f64> = (0..42).map(|i| ((i % 6)*10 + (i/6) + 11) as f64).collect();
	let x  = nn.create_neuron("x", Tensor::<f64>::from_vector(vec![6,7],xs));
	let transpose_x = nn.transpose(Rc::clone(&x));
	let tx = Tensor::<f64>::from_vector(vec![7,6],txs);

	/*
	println!("{}", x.borrow());
    println!("{}", transpose_x.borrow());
	println!("{}", tx);
	 */
	assert_eq!(transpose_x.borrow().ref_signal(), &tx);
    Ok(())
}

#[test]
fn sum_to_test() -> Result<(),Box<dyn std::error::Error>> {
	let mut nn = NeuralNetwork::<f64>::new();
	let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[2,3],&[1.0,2.0,3.0,4.0,5.0,6.0]));
	let y = nn.sum_to(Rc::clone(&x),vec![1,3]);

	assert_eq!(y.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,3],&[5.0,7.0,9.0]));
	nn.backward_propagating(0)?;
	{
		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}
	}

	Ok(())
}

#[test]
fn broadcast_to_test_add() -> Result<(),Box<dyn std::error::Error>> {
	{
		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[2,3],&[1.0,2.0,3.0,4.0,5.0,6.0]));
		let a = nn.create_neuron("a", Tensor::<f64>::from_array(&[1,1],&[10.0]));
		let y = nn.add(Rc::clone(&a),Rc::clone(&x));

		println!("y {}", y.borrow());
		nn.backward_propagating(0)?;

		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			println!("gx {}",gx.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}

		let borrowed_a = a.borrow();
		if let Some(ref ga) = borrowed_a.ref_grad() {
			println!("ga {}",ga.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}
	}

	{
		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[2,3],&[1.0,2.0,3.0,4.0,5.0,6.0]));
		let a = nn.create_neuron("a", Tensor::<f64>::from_array(&[1,1],&[10.0]));
		let y = nn.add(Rc::clone(&x),Rc::clone(&a));

		println!("y {}", y.borrow());
		nn.backward_propagating(0)?;

		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			println!("gx {}",gx.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}

		let borrowed_a = a.borrow();
		if let Some(ref ga) = borrowed_a.ref_grad() {
			println!("ga {}",ga.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}
	}

	{
		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[2,3],&[11.0,12.0,13.0,14.0,15.0,16.0]));
		let y = nn.create_neuron("y", Tensor::<f64>::from_array(&[1,1],&[10.0]));
		let z = nn.sub(Rc::clone(&x),Rc::clone(&y));

		println!("z {}", z.borrow());

		nn.backward_propagating(0)?;
		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			println!("gx {}",gx.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}

		let borrowed_y = y.borrow();
		if let Some(ref gy) = borrowed_y.ref_grad() {
			println!("gy {}",gy.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}
	}

	{
		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[2,3],&[11.0,12.0,13.0,14.0,15.0,16.0]));
		let y = nn.create_neuron("y", Tensor::<f64>::from_array(&[1,1],&[10.0]));
		let z = nn.sub(Rc::clone(&y),Rc::clone(&x));

		println!("z {}", z.borrow());

		nn.backward_propagating(0)?;
		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			println!("gx {}",gx.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}

		let borrowed_y = y.borrow();
		if let Some(ref gy) = borrowed_y.ref_grad() {
			println!("gy {}",gy.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}
	}

	Ok(())
}

#[test]
fn broadcast_to_test_mul() -> Result<(),Box<dyn std::error::Error>> {
	let mut nn = NeuralNetwork::<f64>::new();
	let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[2,3],&[11.0,12.0,13.0,14.0,15.0,16.0]));
	let y = nn.create_neuron("y", Tensor::<f64>::from_array(&[1,1],&[10.0]));
	let z = nn.hadamard_product(Rc::clone(&x),Rc::clone(&y));

	println!("z {}", z.borrow());

	nn.backward_propagating(0)?;

	let borrowed_x = x.borrow();
	if let Some(ref gx) = borrowed_x.ref_grad() {
		println!("gx {}",gx.borrow().ref_signal());
		//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
	}
	else {
		assert!(false);
	}

	let borrowed_y = y.borrow();
	if let Some(ref gy) = borrowed_y.ref_grad() {
		println!("gy {}",gy.borrow().ref_signal());
		//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
	}
	else {
		assert!(false);
	}

	Ok(())
}

#[test]
fn broadcast_to_test_pow() -> Result<(),Box<dyn std::error::Error>> {
	Ok(())
}

#[test]
fn broadcast_to_test_div() -> Result<(),Box<dyn std::error::Error>> {
	{
		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[2,3],&[11.0,12.0,13.0,14.0,15.0,16.0]));
		let y = nn.create_neuron("y", Tensor::<f64>::from_array(&[1,1],&[10.0]));
		let z = nn.hadamard_division(Rc::clone(&x),Rc::clone(&y));

		println!("z {}", z.borrow());

		nn.backward_propagating(0)?;

		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			println!("gx {}",gx.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}

		let borrowed_y = y.borrow();
		if let Some(ref gy) = borrowed_y.ref_grad() {
			println!("gy {}",gy.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}
	}

	{
		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[2,3],&[11.0,12.0,13.0,14.0,15.0,16.0]));
		let y = nn.create_neuron("y", Tensor::<f64>::from_array(&[1,1],&[10.0]));
		let z = nn.hadamard_division(Rc::clone(&y),Rc::clone(&x));

		println!("z {}", z.borrow());

		nn.backward_propagating(0)?;

		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			println!("gx {}",gx.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}

		let borrowed_y = y.borrow();
		if let Some(ref gy) = borrowed_y.ref_grad() {
			println!("gy {}",gy.borrow().ref_signal());
			//assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::one(&[2,3]));
		}
		else {
			assert!(false);
		}
	}

	Ok(())
}


#[test]
fn matmul_test() -> Result<(),Box<dyn std::error::Error>> {

	{
		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[2,3],&[1.0,2.0,3.0,4.0,5.0,6.0]));
		let y = nn.create_neuron("y", Tensor::<f64>::from_array(&[3,2],&[1.0,2.0,3.0,4.0,5.0,6.0]));
		let z = nn.matrix_product(Rc::clone(&x), Rc::clone(&y));

		//println!("z {}", z.borrow());
		assert_eq!(z.borrow().ref_signal(), &Tensor::<f64>::from_array(&[2,2], &[22.0,28.0,49.0,64.0]));
		nn.backward_propagating(0)?;

		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			//println!("gx {}",gx.borrow().ref_signal());
			assert_eq!(gx.borrow().ref_signal(), &Tensor::<f64>::from_array(&[2,3], &[3.0,7.0,11.0,3.0,7.0,11.0]));
		}
		else {
			assert!(false);
		}

		let borrowed_y = y.borrow();
		if let Some(ref gy) = borrowed_y.ref_grad() {
			//println!("gy {}",gy.borrow().ref_signal());
			assert_eq!(gy.borrow().ref_signal(), &Tensor::<f64>::from_array(&[3,2], &[5.0,5.0,7.0,7.0,9.0,9.0]));
		}
		else {
			assert!(false);
		}
	}

	{
		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,6],&[1.0,2.0,3.0,4.0,5.0,6.0]));
		let y = nn.create_neuron("y", Tensor::<f64>::from_array(&[6,1],&[1.0,2.0,3.0,4.0,5.0,6.0]));
		let z = nn.matrix_product(Rc::clone(&x), Rc::clone(&y));

		println!("z {}", z.borrow());
		assert_eq!(z.borrow().ref_signal(), &Tensor::<f64>::from_array(&[1,1], &[91.0]));
		nn.backward_propagating(0)?;

		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			println!("gx {}",gx.borrow().ref_signal());
			assert_eq!(gx.borrow().ref_signal(), borrowed_x.ref_signal());
		}
		else {
			assert!(false);
		}

		let borrowed_y = y.borrow();
		if let Some(ref gy) = borrowed_y.ref_grad() {
			println!("gy {}",gy.borrow().ref_signal());
			assert_eq!(gy.borrow().ref_signal(), borrowed_y.ref_signal());
		}
		else {
			assert!(false);
		}
	}

	Ok(())
}

#[test]
fn sigmoid_test() -> Result<(),Box<dyn std::error::Error>> {
	{
		let mut nn = NeuralNetwork::<f64>::new();
		let xs:Vec<f64> = vec![1.0,2.0,3.0,4.0,5.0,6.0];
		let x = nn.create_neuron("x", Tensor::<f64>::from_vector(vec![2,3],xs.clone()));
		let y = nn.sigmoid(Rc::clone(&x));

		fn sigmoid(x:&Tensor<f64>) -> Tensor<f64> {
			Tensor::<f64>::sigmoid(x)
		}

		println!("y {}", y.borrow());
		let x_sigmoids = xs.iter().map(|x| 1.0/(1.0+(-x).exp())).collect::<Vec<f64>>();
		{
			let borrowed_y =  y.borrow();
			let ys = borrowed_y.ref_signal().buffer().clone();
			assert_eq!(ys,x_sigmoids);
		}

		nn.backward_propagating(0)?;

		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			let diff = differntial(&sigmoid,
								   borrowed_x.ref_signal(),
								   0.001);
			let error = (diff - gx.borrow().ref_signal()).abs();
			assert!(error.buffer().iter().fold(false,|b, &e| { b | (e < 0.0001)}));
		}
		else {
			assert!(false);
		}
	}
	{
		let mut rng = XorShiftRng::from_entropy();
		let uniform_dist = Uniform::new(-6.0,6.0);
		let mut nn = NeuralNetwork::<f64>::new();
		let xs:Vec<f64> = (0..1000).map(|_| uniform_dist.sample(&mut rng)).collect();
		let x = nn.create_neuron("x", Tensor::<f64>::from_vector(vec![1,1000],xs.clone()));
		let y = nn.sigmoid(Rc::clone(&x));

		let borrowed_y = y.borrow();
		let data_points:Vec<(&f64,&f64)> = xs.iter().zip(borrowed_y.ref_signal().buffer().into_iter()).collect();
		let render_backend = BitMapBackend::new("sigmoid_graph.png", (640, 480)).into_drawing_area();
		render_backend.fill(&WHITE);
		let mut chart_builder = ChartBuilder::on(&render_backend)
			.caption("sigmoid(x)", ("sans-serif", 40).into_font())
			.margin(5)
			.x_label_area_size(30)
			.y_label_area_size(30)
			.build_cartesian_2d(-6.0f32..6.0f32, 0.0f32..1.0f32)?;
		chart_builder.configure_mesh().draw()?;
		chart_builder.draw_series(data_points.iter().map(|(x,y)| Circle::new((**x as f32,**y as f32), 2, GREEN.filled())))?;
		/*
		chart_builder
			.configure_series_labels()
			.position(SeriesLabelPosition::UpperLeft)
			.background_style(&WHITE.mix(0.8))
			.border_style(&BLACK)
			.draw()?;
*/
		render_backend.present()?;
	}
	Ok(())
}

#[test]
fn affine_test() -> Result<(),Box<dyn std::error::Error>> {
	{
		let mut rng = XorShiftRng::from_entropy();
		let uniform_dist = Uniform::new(-1.0,1.0);
		let normal_dist = Normal::new(0.0,1.0)?;
		let xs:Vec<f64> = (0..100).map(|_| uniform_dist.sample(&mut rng)).collect();
		let ws:Vec<f64> = (0..10).map(|_| normal_dist.sample(&mut rng)).collect();
		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", Tensor::<f64>::from_vector(vec![100,1],xs));
		let w = nn.create_neuron("w", Tensor::<f64>::from_vector(vec![1,10],ws));
		let b = nn.create_neuron("b", Tensor::<f64>::from_array(&[ 1,1],&[uniform_dist.sample(&mut rng)]));

		let y = nn.affine(Rc::clone(&x),Rc::clone(&w),Some(Rc::clone(&b)));

		fn affine(x:&Tensor<f64>, w:&Tensor<f64>, b:&Tensor<f64>) -> Tensor<f64> {
			let xw = Tensor::<f64>::matrix_product(x,w);
			let expand_b = b.broadcast(xw.shape());
			xw+expand_b
		}

		{
			let borrowed_x = x.borrow();
			let borrowed_w = w.borrow();
			let borrowed_b = b.borrow();
			let borrowed_y = y.borrow();
			assert_eq!(borrowed_y.ref_signal(),
					   &affine(borrowed_x.ref_signal(), borrowed_w.ref_signal(), borrowed_b.ref_signal()));
		}
		nn.backward_propagating(0)?;

		{
			let borrowed_x = x.borrow();
			if let Some(ref gx) = borrowed_x.ref_grad() {
				let borrowed_w = w.borrow();
				let borrowed_b = b.borrow();
				let affine_x = |x:&Tensor<f64>| -> Tensor<f64> {
					affine(x, borrowed_w.ref_signal(), borrowed_b.ref_signal())
				};
				let delta = 0.001;
				let diff = differntial(&affine_x, borrowed_x.ref_signal(), delta);
				let error = (diff - gx.borrow().ref_signal()).abs();
				assert!(error.buffer().iter().fold(false,|b, &e| { b | (e < 0.0001)}));
			}
			else {
				assert!(false);
			}
		}

		{
			let borrowed_w = w.borrow();
			if let Some(ref gw) = borrowed_w.ref_grad() {
				let borrowed_x = x.borrow();
				let borrowed_b = b.borrow();
				let affine_w = |w:&Tensor<f64>| -> Tensor<f64> {
					affine(borrowed_x.ref_signal(), w, borrowed_b.ref_signal())
				};
				let delta = 0.001;
				let diff = differntial(&affine_w, borrowed_w.ref_signal(), delta);
				let diff = diff.sum(borrowed_w.shape());
				let error = (diff - gw.borrow().ref_signal()).abs();
				assert!(error.buffer().iter().fold(false,|b, &e| { b | (e < 0.0001)}));
			}
			else {
				assert!(false);
			}
		}

		{
			let borrowed_b = b.borrow();
			if let Some(ref gb) = borrowed_b.ref_grad() {
				let borrowed_x = x.borrow();
				let borrowed_w = w.borrow();
				let affine_b = |b:&Tensor<f64>| -> Tensor<f64> {
					affine(borrowed_x.ref_signal(), borrowed_w.ref_signal(), b)
				};
				let delta = 0.001;
				let diff = differntial(&affine_b, borrowed_b.ref_signal(), delta);
				let diff = diff.sum(borrowed_b.shape());
				let error = (diff - gb.borrow().ref_signal()).abs();
				assert!(error.buffer().iter().fold(false,|b, &e| { b | (e < 0.0001)}));
			}
			else {
				assert!(false);
			}
		}
	}
	Ok(())
}

#[test]
fn mean_square_error_test() -> Result<(),Box<dyn std::error::Error>> {

	{
		let mut nn = NeuralNetwork::<f64>::new();
		let x0 = nn.create_neuron("x0", Tensor::<f64>::from_array(&[1,3],&[0.0,1.0,2.0]));
		let x1 = nn.create_constant("x1", Tensor::<f64>::from_array(&[1,3],&[0.0,1.0,2.0]));

		let e = {
			let borrowed_x0 = x0.borrow();
			let borrowed_x1 = x1.borrow();
			let x0_signal = borrowed_x0.ref_signal();
			let x1_signal = borrowed_x1.ref_signal();
			(x0_signal-x1_signal).pow(2.0).sum(&[1,1]).scale(1.0/(x0_signal.num_of_elements() as f64))
		};
		let mse = nn.mean_square_error(Rc::clone(&x0),Rc::clone(&x1));
		assert_eq!(mse.borrow().ref_signal(), &e);
	}
	{
		let mut rng = XorShiftRng::from_entropy();
		let uniform_dist = Uniform::new(-1.0,1.0);

		let mut nn = NeuralNetwork::<f64>::new();
		let x0 = nn.create_neuron("x0", Tensor::<f64>::from_vector(vec![1,100],
																   (0..100).map(|_| uniform_dist.sample(&mut rng)).collect::<Vec<f64>>()));
		let x1 = nn.create_constant("x1", Tensor::<f64>::from_vector(vec![1,100],
																	 (0..100).map(|_| uniform_dist.sample(&mut rng)).collect::<Vec<f64>>()));

		let e = {
			let borrowed_x0 = x0.borrow();
			let borrowed_x1 = x1.borrow();
			let x0_signal = borrowed_x0.ref_signal();
			let x1_signal = borrowed_x1.ref_signal();
			(x0_signal-x1_signal).pow(2.0).sum(&[1,1]).scale(1.0/(x0_signal.num_of_elements() as f64))
		};
		let mse = nn.mean_square_error(Rc::clone(&x0),Rc::clone(&x1));
		assert_eq!(mse.borrow().ref_signal(), &e);

		nn.backward_propagating(0)?;

		let mse_x0 = |x0:&Tensor::<f64>| -> Tensor::<f64> {
			let borrowed_x1 = x1.borrow();
			let x1_signal = borrowed_x1.ref_signal();
			(x0 - x1_signal).pow(2.0).sum(&[1,1]).scale(1.0/(x0.num_of_elements() as f64))
		};

		let borrowed_x0 = x0.borrow();
		if let Some(ref gx0) = borrowed_x0.ref_grad() {
			let delta:f64 = 1.0e-5;
			let diff_mes_x0 = differntial(&mse_x0,borrowed_x0.ref_signal(),delta);
			let error = (diff_mes_x0 - gx0.borrow().ref_signal()).abs();
			assert!(error.buffer().iter().fold(false,|b, &e| { b | (e < 0.00001)}));
		}
		else {
			return Err(Box::new(MyError::StringMsg("no grad".to_string())));
		}

	}
	Ok(())
}
