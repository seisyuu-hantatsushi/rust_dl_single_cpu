/* -*- tab-width:4 -*- */

use std::fmt::Display;
use std::rc::Rc;
use std::error::Error;

use rand::SeedableRng;
use rand_distr::{Uniform,Distribution};
use rand_pcg::Pcg64;

use deep_learning::neural_network::NeuralNetwork;
use linear_transform::Tensor;

use plotters::prelude::full_palette::*;
use plotters::prelude::{Circle,BitMapBackend,LineSeries,ChartBuilder,PathElement,SeriesLabelPosition};
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
fn linear_layer_test() ->  Result<(),Box<dyn std::error::Error>> {

	let mut rng = Pcg64::from_entropy();
	let mut nn = NeuralNetwork::<f64>::new();
	let layer1 = nn.create_linear_layer("LL1", 10, true);
	let layer2 = nn.create_linear_layer("LL2",  1, true);

	let uniform_dist = Uniform::new(0.0,1.0);
	let xs:Vec<f64> = (0..100).map(|_| uniform_dist.sample(&mut rng)).collect();
	let uniform_dist = Uniform::new(-0.5,0.5);
	let ys:Vec<f64> = xs.iter().map(|&x| (2.0*std::f64::consts::PI*x).sin()+uniform_dist.sample(&mut rng)).collect();
	let x = nn.create_constant("x", Tensor::<f64>::from_vector(vec![100,1], xs));
	let y = nn.create_constant("y", Tensor::<f64>::from_vector(vec![100,1], ys));
	let ll1_output = nn.layer_set_inputs(&layer1, vec![Rc::clone(&x)]);
	let act_output = nn.sigmoid(Rc::clone(&ll1_output[0]));
	let pred_y     = nn.layer_set_inputs(&layer2, vec![Rc::clone(&act_output)]);
	let loss       = nn.mean_square_error(Rc::clone(&y), Rc::clone(&pred_y[0]));
	let learning_rate = 0.2;

	nn.backward_propagating(0)?;

	nn.make_dot_graph(0,"2layer_graph.dot")?;

	for i in 0..10000 {

		for l in vec![&layer1, &layer2] {
			for param in l.get_layer_param() {
				let feedback = if let Some(ref g) = param.borrow().ref_grad() {
					g.borrow().ref_signal().scale(learning_rate)
				}
				else {
					return Err(Box::new(MyError::StringMsg("param does not have grad".to_string())));
				};
				let update = param.borrow().ref_signal() - feedback;
				param.borrow_mut().assign(update);
			}
		}

		if i % 1000 == 0 {
			println!("{}", loss.borrow());
		}

		nn.forward_propagating(0)?;
		nn.forward_propagating(1)?;

	}

	{
		let xt = x.borrow().ref_signal().clone();
		let yt = y.borrow().ref_signal().clone();
		let data_points: Vec<(&f64, &f64)> =
			xt.buffer().iter().zip(yt.buffer().iter()).collect();

		let pred_xs:Vec<f64> = (0..100).map(|i| (i as f64)/100.0).collect();
		let pred_xt = Tensor::<f64>::from_vector(vec![100,1], pred_xs);

		x.borrow_mut().assign(pred_xt.clone());
		nn.forward_propagating(0)?;
		let pred_yt = pred_y[0].borrow().ref_signal().clone();
		let pred_points: Vec<(&f64, &f64)> =
			pred_xt.buffer().iter().zip(pred_yt.buffer().iter()).collect();

		let render_backend = BitMapBackend::new("simple_nn_layer.png", (800,600)).into_drawing_area();
		render_backend.fill(&WHITE)?;
		let mut data_points_chart_builder = ChartBuilder::on(&render_backend)
			.caption("simple neural network (layer)", ("sans-serif", 50).into_font())
			.margin(10)
			.x_label_area_size(30)
			.y_label_area_size(30)
			.build_cartesian_2d(0.0f32..1.0f32, -2.0f32..2.0f32)?;
		data_points_chart_builder.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;

		data_points_chart_builder
			.draw_series(data_points.iter().map(|(x,y)| Circle::new((**x as f32,**y as f32), 2, GREEN.filled())))?;

		data_points_chart_builder
			.draw_series(LineSeries::new(pred_points.iter().map(|(x, y)| (**x as f32, **y as f32)).collect::<Vec<(f32,f32)>>(),&RED))?
			.label(format!("prediction"))
			.legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], &RED));

		data_points_chart_builder
			.configure_series_labels()
			.position(SeriesLabelPosition::UpperLeft)
			.background_style(&WHITE.mix(0.8))
			.border_style(&BLACK)
			.draw()?;
		render_backend.present()?;
	}
	Ok(())
}
