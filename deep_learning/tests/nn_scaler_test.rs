
use deep_learning::neural_network::NeuralNetwork;
use linear_transform::Tensor;
use std::rc::Rc;

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
    let c12 = nn.create_constant("14.0", Tensor::<f64>::from_array(&[1,1],&[12.0]));
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
