/* -*- tab-width:4 -*- */

use std::fmt::Display;
use std::rc::Rc;
use std::error::Error;

use deep_learning::neural_network::NeuralNetwork;
use deep_learning::datasets::*;
use linear_transform::{Tensor,SubTensor};

use plotters::prelude::{BitMapBackend,ChartBuilder};
use plotters::prelude::{Circle,Cross,Rectangle,TriangleMarker,EmptyElement,PathElement,
						SeriesLabelPosition};
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
fn spiral_classification_test() -> Result<(),Box<dyn std::error::Error>> {
	let num_of_class = 3;
	let (xs,ts) = spiral::get_2d_dataset::<f64>(num_of_class,100);
	//println!("{}",xs);
	//println!("{}",ts);
	let render_backend = BitMapBackend::new("spiral_graph.png", (640, 480)).into_drawing_area();
	render_backend.fill(&WHITE)?;

	let mut chart_builder = ChartBuilder::on(&render_backend)
		.caption("spiral", ("sans-serif", 40).into_font())
		.margin(5)
		.x_label_area_size(30)
		.y_label_area_size(30)
		.build_cartesian_2d(-1.56f32..1.56f32, -1.2f32..1.2f32)?;
	chart_builder.configure_mesh().disable_x_mesh().disable_y_mesh().draw()?;

	let class_iter =
			xs.iter().zip(ts.iter()).filter(|(xst,t)| t[vec![0,0]] == 0.0 as f64);
	chart_builder.draw_series(class_iter.map(|(xst,t)| {
		let x = xst[vec![0,0]] as f32;
		let y = xst[vec![0,1]] as f32;
		EmptyElement::at((x,y)) + Circle::new((0,0), 2, GREEN.filled())
	}))?;
	let class_iter =
			xs.iter().zip(ts.iter()).filter(|(xst,t)| t[vec![0,0]] == 1.0 as f64);
	chart_builder.draw_series(class_iter.map(|(xst,t)| {
		let x = xst[vec![0,0]] as f32;
		let y = xst[vec![0,1]] as f32;
		EmptyElement::at((x,y)) + TriangleMarker::new((0,0), 2, RED.filled())
	}))?;
	let class_iter =
		xs.iter().zip(ts.iter()).filter(|(xst,t)| t[vec![0,0]] == 2.0 as f64);
	chart_builder.draw_series(class_iter.map(|(xst,t)| {
		let x = xst[vec![0,0]] as f32;
		let y = xst[vec![0,1]] as f32;
		EmptyElement::at((x,y)) + Cross::new((0,0), 2, CYAN.filled())
	}))?;
	/*
	let class_iter =
		xs.iter().zip(ts.iter()).filter(|(xst,t)| t[vec![0,0]] == 3.0 as f64);
	chart_builder.draw_series(class_iter.map(|(xst,t)| {
		let x = xst[vec![0,0]] as f32;
		let y = xst[vec![0,1]] as f32;
		EmptyElement::at((x,y)) + Rectangle::new([(0,0),(2,2)], BLACK.filled())
	}))?;
*/
	render_backend.present()?;

	{
		
	}

	Ok(())
}
