/* -*- tab-width:4 -*- */

use std::fmt::Display;
use std::rc::Rc;
use std::error::Error;

use deep_learning::neural_network::NeuralNetwork;
use linear_transform::Tensor;

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
		let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[2,3],&[1.0,2.0,3.0,4.0,5.0,6.0]));
		let y = nn.sigmod(Rc::clone(&x));

		println!("y {}", y.borrow());

		nn.backward_propagating(0)?;

		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			println!("gx {}",gx.borrow().ref_signal());
			let x_t = Tensor::<f64>::from_array(&[2,3],&[1.0,2.0,3.0,4.0,5.0,6.0]);
			let one = Tensor::<f64>::one(&[2,3]);
			let gx_t = Tensor::<f64>::hadamard_product(&x_t, &(&one - &x_t));
			assert_eq!(gx.borrow().ref_signal(), &gx_t)
		}
		else {
			assert!(false);
		}
	}
	Ok(())
}

#[test]
fn affine_test() -> Result<(),Box<dyn std::error::Error>> {
	{
		let mut nn = NeuralNetwork::<f64>::new();
		let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[10,1],&[2.0f64;10]));
		let w = nn.create_neuron("w", Tensor::<f64>::from_array(&[1,10],&[3.0f64;10]));
		let b = nn.create_neuron("b", Tensor::<f64>::from_array(&[ 1,1],&[2.0f64;1]));

		let y = nn.affine(Rc::clone(&x),Rc::clone(&w),Rc::clone(&b));

		println!("y {}", y.borrow());

		assert_eq!(y.borrow().ref_signal(),
				   &Tensor::<f64>::from_array(&[10,10],&[8.0f64;100]));

		nn.backward_propagating(0)?;

		let borrowed_x = x.borrow();
		if let Some(ref gx) = borrowed_x.ref_grad() {
			println!("gx {}",gx.borrow().ref_signal());
		}
		else {
			assert!(false);
		}

		let borrowed_w = w.borrow();
		if let Some(ref gw) = borrowed_w.ref_grad() {
			println!("gw {}",gw.borrow().ref_signal());
			
		}
		else {
			assert!(false);
		}

		let borrowed_b = b.borrow();
		if let Some(ref gb) = borrowed_b.ref_grad() {
			println!("gb {}",gb.borrow().ref_signal());
			let gb_t = Tensor::<f64>::from_array(&[1,1], &[100.0]);
			assert_eq!(gb.borrow().ref_signal(), &gb_t)
		}
		else {
			assert!(false);
		}
	}
	Ok(())
}
