/* -*- tab-width:4 -*- */

use std::fmt::Display;
use std::rc::Rc;
use std::error::Error;

use deep_learning::neural_network::NeuralNetwork;
use linear_transform::Tensor;
use plotters::prelude::{BitMapBackend,LineSeries,ChartBuilder,PathElement};
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

#[test]
#[ignore]
fn high_order_diff() {
	let mut nn = NeuralNetwork::<f64>::new();
	let x0 = 2.0;
	let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,1],&[x0]));
	let c4 = nn.create_constant("4.0", Tensor::<f64>::from_array(&[1,1],&[4.0]));
	let y = nn.pow_rank0(Rc::clone(&x),c4);
	y.borrow_mut().rename("y");

	let _ = nn.clear_grads(0);
	match nn.backward_propagating(0) {
		Ok(outputs) => {
			for output in outputs.iter() {
				println!("gx1 {}",output.borrow());
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}

	let _ = nn.clear_grads(0);
	let _ = nn.clear_grads(1);
	match nn.backward_propagating(1) {
		Ok(outputs) => {
			for output in outputs.iter() {
				println!("gx2 {}",output.borrow());
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}

	let _ = nn.clear_grads(0);
	let _ = nn.clear_grads(1);
	let _ = nn.clear_grads(2);
	match nn.backward_propagating(2) {
		Ok(outputs) => {
			for output in outputs.iter() {
				println!("gx3 {}",output.borrow());
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}

	let _ = nn.clear_grads(0);
	let _ = nn.clear_grads(1);
	let _ = nn.clear_grads(2);
	let _ = nn.clear_grads(3);
	match nn.backward_propagating(3) {
		Ok(outputs) => {
			for output in outputs.iter() {
				println!("gx4 {}",output.borrow());
S			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}

	let _ = nn.clear_grads(0);
	let _ = nn.clear_grads(1);
	let _ = nn.clear_grads(2);
	let _ = nn.clear_grads(3);
	let _ = nn.clear_grads(4);
	match nn.backward_propagating(4) {
		Ok(outputs) => {
			for output in outputs.iter() {
				println!("gx5 {}",output.borrow());
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}

	let _ = nn.clear_grads(0);
	let _ = nn.clear_grads(1);
	let _ = nn.clear_grads(2);
	let _ = nn.clear_grads(3);
	let _ = nn.clear_grads(4);
	let _ = nn.clear_grads(5);
	match nn.backward_propagating(5) {
		Ok(outputs) => {
			for output in outputs.iter() {
				println!("gx6 {}",output.borrow());
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}

	let _ = nn.clear_grads(0);
	let _ = nn.clear_grads(1);
	let _ = nn.clear_grads(2);
	let _ = nn.clear_grads(3);
	let _ = nn.clear_grads(4);
	let _ = nn.clear_grads(5);
	let _ = nn.clear_grads(6);
	match nn.backward_propagating(6) {
		Ok(outputs) => {
			for output in outputs.iter() {
				println!("gx7 {}",output.borrow());
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}

	if let Err(e) = nn.make_dot_graph(0,"graph2_order0.dot") {
		println!("{}",e);
		assert!(false)
	}

	if let Err(e) = nn.make_dot_graph(1,"graph2_order1.dot") {
		println!("{}",e);
		assert!(false)
	}

	if let Err(e) = nn.make_dot_graph(2,"graph2_order2.dot") {
		println!("{}",e);
		assert!(false)
	}

	if let Err(e) = nn.make_dot_graph(3,"graph2_order3.dot") {
		println!("{}",e);
		assert!(false)
	}

	if let Err(e) = nn.make_dot_graph(4,"graph2_order4.dot") {
		println!("{}",e);
		assert!(false)
	}
}

#[test]
#[ignore]
fn sphere_test(){
	//sphere
	let mut nn = NeuralNetwork::<f64>::new();
	let (x0,y0) = (1.0,1.0);
	let x  = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,1],&[x0]));
	let y  = nn.create_neuron("y", Tensor::<f64>::from_array(&[1,1],&[y0]));
	let c2 = nn.create_constant("2.0", Tensor::<f64>::from_array(&[1,1],&[2.0]));
	fn sphere(x:f64,y:f64) -> f64 {
		x.powf(2.0) + y.powf(2.0)
	}

	let term1 = nn.pow_rank0(Rc::clone(&x),Rc::clone(&c2));
	let term2 = nn.pow_rank0(Rc::clone(&y),Rc::clone(&c2));
	let z = nn.add(term1, term2);
	z.borrow_mut().rename("z");

	println!("z {}\n{}",sphere(x0,y0), z.borrow());

	match nn.backward_propagating(0) {
		Ok(_outputs) => {
			let borrowed_x = x.borrow();
			if let Some(ref g) = borrowed_x.ref_grad() {
				let delta = 1.0e-5;
				let diff = (sphere(x0+delta,y0)-sphere(x0-delta,y0))/(2.0*delta);
				println!("gx {} {}", g.borrow(), diff);
				let gx = g.borrow().ref_signal()[vec![0,0]];
				assert!((diff-gx).abs() <= delta);
			}
			else {
				assert!(false);
			};
			let borrowed_y = y.borrow();
			if let Some(ref g) = borrowed_y.ref_grad() {
				let delta = 1.0e-5;
				let diff = (sphere(x0,y0+delta)-sphere(x0,y0-delta))/(2.0*delta);
				println!("gy {} {}", g.borrow(), diff);
				let gy = g.borrow().ref_signal()[vec![0,0]];
				assert!((diff-gy).abs() <= delta);
			}
			else {
				assert!(false);
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}
}

#[test]
#[ignore]
fn matyas_test(){
	//matyas
	let mut nn = NeuralNetwork::<f64>::new();
	let (x0,y0) = (1.0,1.0);
	let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,1],&[x0]));
	let y = nn.create_neuron("y", Tensor::<f64>::from_array(&[1,1],&[y0]));
	let c2 = nn.create_constant("2.0", Tensor::<f64>::from_array(&[1,1],&[2.0]));
	let c48 = nn.create_constant("0.48", Tensor::<f64>::from_array(&[1,1],&[0.48]));
	let c26 = nn.create_constant("0.26", Tensor::<f64>::from_array(&[1,1],&[0.26]));
	fn matyas(x:f64,y:f64) -> f64 {
		0.26*(x.powf(2.0) + y.powf(2.0)) - 0.48*x*y
	}

	let term1 = nn.mul_rank0(Rc::clone(&x),Rc::clone(&y));
	let term2 = nn.mul_rank0(Rc::clone(&c48),term1);
	let term3 = nn.pow_rank0(Rc::clone(&x),Rc::clone(&c2));
	let term4 = nn.pow_rank0(Rc::clone(&y),Rc::clone(&c2));
	let term5 = nn.add(term3,term4);
	let term  = nn.mul_rank0(Rc::clone(&c26),term5);
	let z = nn.sub(term,term2);
	z.borrow_mut().rename("z");

	println!("matyas z {}\n{}",matyas(x0,y0), z.borrow());

	match nn.backward_propagating(0) {
		Ok(_outputs) => {
			let borrowed_x = x.borrow();
			if let Some(ref g) = borrowed_x.ref_grad() {
				let delta = 1.0e-6;
				let diff = (matyas(x0+delta,y0)-matyas(x0-delta,y0))/(2.0*delta);
				println!("matyas gx {} {}", g.borrow(), diff);
				let gx = g.borrow().ref_signal()[vec![0,0]];
				assert!((diff-gx).abs() <= delta);
			}
			else {
				assert!(false);
			};
			let borrowed_y = y.borrow();
			if let Some(ref g) = borrowed_y.ref_grad() {
				let delta = 1.0e-6;
				let diff = (matyas(x0,y0+delta)-matyas(x0,y0-delta))/(2.0*delta);
				println!("matyas gy {} {}", g.borrow(), diff);
				let gy = g.borrow().ref_signal()[vec![0,0]];
				assert!((diff-gy).abs() <= delta);
			}
			else {
				assert!(false);
			}
				},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}
	println!("");
}

#[test]
fn goldstein_price_test() {
    //Goldstein-Price
    let mut nn = NeuralNetwork::<f64>::new();
    let (x0,y0) = (1.0,1.0);
    let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,1],&[x0]));
    let y = nn.create_neuron("y", Tensor::<f64>::from_array(&[1,1],&[y0]));
    let c1  = nn.create_constant("1.0",  Tensor::<f64>::from_array(&[1,1],&[1.0]));
    let c2  = nn.create_constant("2.0",  Tensor::<f64>::from_array(&[1,1],&[2.0]));
    let c3  = nn.create_constant("3.0",  Tensor::<f64>::from_array(&[1,1],&[3.0]));
    let c6  = nn.create_constant("6.0",  Tensor::<f64>::from_array(&[1,1],&[6.0]));
    let c12 = nn.create_constant("12.0", Tensor::<f64>::from_array(&[1,1],&[12.0]));
    let c14 = nn.create_constant("14.0", Tensor::<f64>::from_array(&[1,1],&[14.0]));
    let c18 = nn.create_constant("18.0", Tensor::<f64>::from_array(&[1,1],&[18.0]));
    let c19 = nn.create_constant("19.0", Tensor::<f64>::from_array(&[1,1],&[19.0]));
    let c27 = nn.create_constant("27.0", Tensor::<f64>::from_array(&[1,1],&[27.0]));
    let c30 = nn.create_constant("30.0", Tensor::<f64>::from_array(&[1,1],&[30.0]));
    let c32 = nn.create_constant("32.0", Tensor::<f64>::from_array(&[1,1],&[32.0]));
    let c36 = nn.create_constant("36.0", Tensor::<f64>::from_array(&[1,1],&[36.0]));
    let c48 = nn.create_constant("48.0", Tensor::<f64>::from_array(&[1,1],&[48.0]));

    fn goldstein_price(x:f64,y:f64) -> f64 {
		(1.0 + (x + y + 1.0).powf(2.0)*(19.0 - 14.0*x + 3.0*x.powf(2.0) - 14.0*y + 6.0*x*y + 3.0*y.powf(2.0)))*(30.0 + (2.0*x-3.0*y).powf(2.0)*(18.0 - 32.0*x + 12.0*x.powf(2.0) + 48.0*y - 36.0*x*y + 27.0*y.powf(2.0)))
    }

    let term   = nn.add(Rc::clone(&x),Rc::clone(&y));
    let term1  = nn.add(Rc::clone(&c1),term);
    let term2  = nn.pow_rank0(term1,Rc::clone(&c2));
    let term3  = nn.mul_rank0(Rc::clone(&c14), Rc::clone(&x));
    let term   = nn.sub(Rc::clone(&c19),term3);
    let term1  = nn.pow_rank0(Rc::clone(&x), Rc::clone(&c2)); //x^2.0
    let term3  = nn.mul_rank0(Rc::clone(&c3),term1); //3.0*(x^2.0)
    let term   = nn.add(term, term3);
    let term1  = nn.mul_rank0(Rc::clone(&c14), Rc::clone(&y));
    let term   = nn.sub(term, term1);
    let term1  = nn.mul_rank0(Rc::clone(&x), Rc::clone(&y));
    let term1  = nn.mul_rank0(Rc::clone(&c6), term1);
    let term   = nn.add(term, term1);
    let term1  = nn.pow_rank0(Rc::clone(&y), Rc::clone(&c2));
    let term1  = nn.mul_rank0(Rc::clone(&c3), term1);
    let term   = nn.add(term, term1);
    let term   = nn.mul_rank0(term2, term);
    let term_l = nn.add(Rc::clone(&c1), term);
    let term1  = nn.mul_rank0(Rc::clone(&c2), Rc::clone(&x));
    let term2  = nn.mul_rank0(Rc::clone(&c3), Rc::clone(&y));
    let term   = nn.sub(term1,term2);
    let term1  = nn.pow_rank0(term, Rc::clone(&c2)); //(2.0*x-3.0*y)^2
    let term   = nn.mul_rank0(Rc::clone(&c32), Rc::clone(&x));
    let term   = nn.sub(c18, term);
    let term2  = nn.pow_rank0(Rc::clone(&x), Rc::clone(&c2));
    let term2  = nn.mul_rank0(Rc::clone(&c12), term2); //12*(x^2)
    let term   = nn.add(term, term2);
    let term2  = nn.mul_rank0(Rc::clone(&c48), Rc::clone(&y));
    let term   = nn.add(term, term2);
    let term2  = nn.mul_rank0(Rc::clone(&x), Rc::clone(&y));
    let term2  = nn.mul_rank0(Rc::clone(&c36), term2);
    let term   = nn.sub(term, term2);
    let term2  = nn.pow_rank0(Rc::clone(&y), Rc::clone(&c2));
    let term2  = nn.mul_rank0(Rc::clone(&c27), term2);
    let term   = nn.add(term, term2);
    let term   = nn.mul_rank0(term1,term);
    let term_r = nn.add(Rc::clone(&c30),term);
    let z = nn.mul_rank0(term_l, term_r);

    z.borrow_mut().rename("z");
    println!("gs {} \n{}", goldstein_price(x0, y0), z.borrow());

	match nn.backward_propagating(0) {
		Ok(_outputs) => {
			let borrowed_x = x.borrow();
			if let Some(ref g) = borrowed_x.ref_grad() {
				let delta = 1.0e-6;
				let diff = (goldstein_price(x0+delta,y0)-goldstein_price(x0-delta,y0))/(2.0*delta);
				println!("gs gx {} {}", g.borrow(), diff);
				let gx = g.borrow().ref_signal()[vec![0,0]];
				assert!((diff-gx).abs() <= delta);
			}
			else {
				assert!(false);
			};
			let borrowed_y = y.borrow();
			if let Some(ref g) = borrowed_y.ref_grad() {
				let delta = 1.0e-6;
				let diff = (goldstein_price(x0,y0+delta)-goldstein_price(x0,y0-delta))/(2.0*delta);
				println!("gs gy {} {}", g.borrow(), diff);
				let gy = g.borrow().ref_signal()[vec![0,0]];
				assert!((diff-gy).abs() <= delta);
			}
			else {
				assert!(false);
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}
}

#[test]
#[ignore]
fn second_order_diff() {
	let mut nn = NeuralNetwork::<f64>::new();
	let x0 = 2.0;
	let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,1],&[x0]));
	let c4 = nn.create_constant("4.0", Tensor::<f64>::from_array(&[1,1],&[4.0]));
	let c2 = nn.create_constant("2.0", Tensor::<f64>::from_array(&[1,1],&[2.0]));
	let term1 = nn.pow_rank0(Rc::clone(&x),c4);
	let term2 = nn.pow_rank0(Rc::clone(&x),Rc::clone(&c2));
	let term3 = nn.mul_rank0(Rc::clone(&c2), term2);
	let y = nn.sub(term1,term3);
	y.borrow_mut().rename("y");
/*
	if let Err(e) = nn.make_dot_graph(0,"graph2.dot") {
	println!("{}",e);
	assert!(false)
}
*/
	fn func(x:f64) -> f64 {
		x.powf(4.0) - 2.0 * x.powf(2.0)
	}
	assert_eq!(y.borrow().ref_signal()[vec![0,0]],func(x0));

	match nn.backward_propagating(0) {
		Ok(outputs) => {
			for output in outputs.iter() {
				let delta = 1.0e-5;
				let diff = (func(x0+delta)-func(x0-delta))/(2.0*delta);
				println!("gx1 {}",output.borrow());
				let gx = output.borrow().ref_signal()[vec![0,0]];
				assert!((diff-gx).abs() <= delta);
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}

	let _ = nn.clear_grads(0);
	let _ = nn.clear_grads(1);

	match nn.backward_propagating(1) {
		Ok(_outputs) => {
			let borrowed_x = x.borrow();
			if let Some(ref g) = borrowed_x.ref_grad() {
				let delta = 1.0e-5;
				let diff2 = (func(x0+delta) + func(x0-delta) - 2.0 * func(x0))/(delta.powf(2.0));
				println!("gx2 {}",g.borrow());
				let gx2 = g.borrow().ref_signal()[vec![0,0]];
				assert!((diff2-gx2).abs() <= delta);
			}
			else {
				assert!(false);
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}
/*
				if let Err(e) = nn.make_dot_graph(0,"graph3_order0.dot") {
					println!("{}",e);
					assert!(false)
				}

				if let Err(e) = nn.make_dot_graph(1,"graph3_order1.dot") {
					println!("{}",e);
					assert!(false)
				}

				if let Err(e) = nn.make_dot_graph(2,"graph3_order2.dot") {
					println!("{}",e);
					assert!(false)
				}
*/
}

#[test]
fn sin_taylor_expansion_test() {
	//Taylor Expansion of Sin
	let mut nn = NeuralNetwork::<f64>::new();
	let x0 = std::f64::consts::PI/2.0;
	let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,1],&[x0]));
	let mut tail_term = Rc::clone(&x);
	let mut counter = 2;
	loop {
		let index_int = (counter-1)*2 + 1;
		let index_real = index_int as f64;
		let index = nn.create_constant(&index_int.to_string(),
									   Tensor::<f64>::from_array(&[1,1],&[index_real]));
		let sign:f64 = (-1.0f64).powi(counter-1);
		let factorial:f64 = sign*(1..(index_int+1)).fold(1.0, |p, n| p * (n as f64));
		let p = nn.pow_rank0(Rc::clone(&x), index);
		let factorial_constant = nn.create_constant(&format!("{}({}!)",sign,index_int),
													Tensor::<f64>::from_array(&[1,1],&[factorial]));
		let term = nn.div_rank0(Rc::clone(&p), factorial_constant);
		let label = "term_".to_string() + &counter.to_string();
		term.borrow_mut().rename(&label);
		tail_term = nn.add(tail_term, Rc::clone(&term));

		let t = term.borrow().ref_signal()[vec![0,0]];
		if t.abs() <= 1.0e-6 {
			break;
		}
		counter += 1;
		if counter >= 10000 {
			break;
		}
	}
	{
		let diff = (tail_term.borrow().ref_signal()[vec![0,0]]-x0.sin()).abs();
		println!("{} {}",
				 tail_term.borrow().ref_signal()[vec![0,0]],
						 x0.sin());
		assert!(diff < 1.0e-5);
	}

	match nn.backward_propagating(0) {
		Ok(_outputs) => {
			let borrowed_x = x.borrow();
			if let Some(ref g) = borrowed_x.ref_grad() {
				let delta = 1.0e-5;
				let diff = ((x0+delta).sin() - (x0-delta).sin())/(2.0*delta);
				println!("gx {} {}",g.borrow(),diff);
			}
			else {
				assert!(false);
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	}

	if let Err(e) = nn.make_dot_graph(0,"sin_taylor_order0.dot") {
		println!("{}",e);
		assert!(false)
	}

	if let Err(e) = nn.make_dot_graph(1,"sin_taylor_order1.dot") {
		println!("{}",e);
		assert!(false)
	}

}

#[test]
fn sin_high_order_diff_test() -> Result<(),Box<dyn std::error::Error>> {

	let mut nn = NeuralNetwork::<f64>::new();
	let xs_src:Vec<f64> = (-400..=400).map(|i| (i as f64)*(7.0/400.0)).collect();
	//println!("{}", xs_src.len());
	let xs = nn.create_neuron("xs",Tensor::<f64>::from_vector(vec![1,xs_src.len()],xs_src));
	//println!("{}", xs.borrow());
	let sin_xs = nn.sin(Rc::clone(&xs));
	//println!("{}", sin_xs.borrow());

	//println!("sin_xs shape {:?}", sin_xs.borrow().shape());

	let render_backend = BitMapBackend::new("sin_graph.png", (640, 480)).into_drawing_area();
	let _ = render_backend.fill(&WHITE);
	let mut chart_builder = ChartBuilder::on(&render_backend)
		.caption("sin(x)", ("sans-serif", 40).into_font())
		.margin(5)
		.x_label_area_size(30)
		.y_label_area_size(30)
		.build_cartesian_2d(-7.0f32..7.0f32, -1.0f32..1.0f32).unwrap();
	chart_builder.configure_mesh().draw()?;

	{
		let (xs_holder, sin_xs_holder) = (xs.borrow(), sin_xs.borrow());
		let raw_xs = xs_holder.ref_signal().buffer();
		let raw_sin_xs = sin_xs_holder.ref_signal().buffer();
		let points = raw_xs.iter().zip(raw_sin_xs.iter());
		chart_builder
			.draw_series(LineSeries::new(
				points.map(|(x, sin_x)| (*x as f32, *sin_x as f32)),
				&RED,
			))?
			.label("y = sin(x)")
			.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
	}
	{
		match nn.backward_propagating(0) {
			Ok(_outputs) => {
				let borrowed_x = xs.borrow();
				if let Some(ref _g) = borrowed_x.ref_grad() {
					//println!("gx {}",g.borrow());
				}
				else {
					return Err(Box::new(MyError::StringMsg("gx None".to_string())));
				}
			},
			Err(e) => {
				println!("{}",e);
				assert!(false)
			}
		};

		let xs_holder = xs.borrow();
		let raw_xs = xs_holder.ref_signal().buffer();
		let grad_xs_holder = if let Some(ref g) = xs_holder.ref_grad() {
			g.borrow()
		}
		else {
			return Err(Box::new(MyError::StringMsg("gx None".to_string())));
		};
		let raw_grad_xs = grad_xs_holder.ref_signal().buffer();
		let points = raw_xs.iter().zip(raw_grad_xs.iter());
		chart_builder
			.draw_series(LineSeries::new(
				points.map(|(x, sin_x)| (*x as f32, *sin_x as f32)),
				&GREEN,
			))?
			.label("y = sin'(x)")
			.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
	}

	{
		let _ = nn.clear_grads(0);
		let _ = nn.clear_grads(1);
		match nn.backward_propagating(1) {
			Ok(_outputs) => {
				let borrowed_x = xs.borrow();
				if let Some(ref _g) = borrowed_x.ref_grad() {
					//println!("gx {}",g.borrow());
				}
				else {
					return Err(Box::new(MyError::StringMsg("gx None".to_string())));
				}
			},
			Err(e) => {
				println!("{}",e);
				assert!(false)
			}
		};

		let xs_holder = xs.borrow();
		let raw_xs = xs_holder.ref_signal().buffer();
		let grad_xs_holder = if let Some(ref g) = xs_holder.ref_grad() {
			g.borrow()
		}
		else {
			return Err(Box::new(MyError::StringMsg("gx None".to_string())));
		};
		let raw_grad_xs = grad_xs_holder.ref_signal().buffer();
		let points = raw_xs.iter().zip(raw_grad_xs.iter());
		chart_builder
			.draw_series(LineSeries::new(
				points.map(|(x, sin_x)| (*x as f32, *sin_x as f32)),
				&CYAN,
			))?
			.label("y = sin''(x)")
			.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &CYAN));
	}

	{
		let _ = nn.clear_grads(0);
		let _ = nn.clear_grads(1);
		let _ = nn.clear_grads(2);
		match nn.backward_propagating(2) {
			Ok(_outputs) => {
				let borrowed_x = xs.borrow();
				if let Some(ref _g) = borrowed_x.ref_grad() {
					//println!("gx {}",g.borrow());
				}
				else {
					return Err(Box::new(MyError::StringMsg("gx None".to_string())));
				}
			},
			Err(e) => {
				println!("{}",e);
				assert!(false)
			}
		};

		let xs_holder = xs.borrow();
		let raw_xs = xs_holder.ref_signal().buffer();
		let grad_xs_holder = if let Some(ref g) = xs_holder.ref_grad() {
			g.borrow()
		}
		else {
			return Err(Box::new(MyError::StringMsg("gx None".to_string())));
		};
		let raw_grad_xs = grad_xs_holder.ref_signal().buffer();
		let points = raw_xs.iter().zip(raw_grad_xs.iter());
		chart_builder
			.draw_series(LineSeries::new(
				points.map(|(x, sin_x)| (*x as f32, *sin_x as f32)),
				&ORANGE_800,
			))?
			.label("y = sin'''(x)")
			.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &ORANGE_800));
	}

	chart_builder
		.configure_series_labels()
		.background_style(&WHITE.mix(0.8))
		.border_style(&BLACK)
		.draw()?;

	render_backend.present()?;
	Ok(())
}

#[test]
#[ignore]
fn tanh_high_order_diff_test() -> Result<(),Box<dyn std::error::Error>> {

	let mut nn = NeuralNetwork::<f64>::new();
	let x0 = 1.0;
	let x = nn.create_neuron("x", Tensor::<f64>::from_array(&[1,1],&[x0]));
	let y = nn.tanh(Rc::clone(&x));

	let diff = 1.0f64.tanh() - y.borrow().ref_signal()[vec![0,0]];
	println!("{} {}", 1.0f64.tanh(), y.borrow());
	assert!(diff.abs() < 1.0e-5);

	match nn.backward_propagating(0) {
		Ok(_outputs) => {
			let borrowed_x = x.borrow();
			if let Some(ref g) = borrowed_x.ref_grad() {
				let gx = g.borrow().ref_signal()[vec![0,0]];
				let delta = 1.0e-6;
				let diff = ((1.0f64+delta).tanh()-(1.0f64-delta).tanh())/(2.0*delta);
				assert!((diff - gx).abs() < delta);
				println!("1st order diff {} {}",gx,diff);
			}
			else {
				return Err(Box::new(MyError::StringMsg("gx None".to_string())));
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	};

	let _ = nn.clear_grads(0);
	let _ = nn.clear_grads(1);
	match nn.backward_propagating(1) {
		Ok(_outputs) => {
			let borrowed_x = x.borrow();
			if let Some(ref g) = borrowed_x.ref_grad() {
				let gx = g.borrow().ref_signal()[vec![0,0]];
				let delta = 1.0e-6;
				let diff = ((1.0f64+delta).tanh()+(1.0f64-delta).tanh()-2.0*1.0f64.tanh())/delta.powf(2.0);
				println!("second order {} {}",gx,diff);
				//assert!((diff - gx).abs() < delta);
			}
			else {
				return Err(Box::new(MyError::StringMsg("gx None".to_string())));
			}
		},
		Err(e) => {
			println!("{}",e);
			assert!(false)
		}
	};

	if let Err(e) = nn.make_dot_graph(0,"tanh_taylor_order0.dot") {
		println!("{}",e);
		assert!(false)
	}

	if let Err(e) = nn.make_dot_graph(1,"tanh_taylor_order1.dot") {
		println!("{}",e);
		assert!(false)
	}

	if let Err(e) = nn.make_dot_graph(2,"tanh_taylor_order2.dot") {
		println!("{}",e);
		assert!(false)
	}
	
	Ok(())
}
