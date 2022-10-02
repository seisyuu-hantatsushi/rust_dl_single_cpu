
use std::path::Path;
use std::collections::HashMap;
use hdf5;
use linear_transform::tensor::tensor_base::Tensor;

pub trait Loader<T>
where T:Clone {
    fn from_hdf5(file:&str) -> Result<HashMap<String,T>,String>;
}

impl<T> Loader<Tensor<T>> for Tensor<T>
where T:num::Float+Clone+hdf5::H5Type
{
    fn from_hdf5(file:&str) -> Result<HashMap<String,Tensor<T>>,String> {
	let datasets_file_path = Path::new(file);
	let mut datasets = HashMap::<String,Tensor<T>>::new();

	if !datasets_file_path.exists() {
            return Err("weight file does not exists".to_string());
	}

	let f = match hdf5::File::open(file) {
	    Ok(f) => f,
	    Err(e) => {
		match e {
		    hdf5::Error::HDF5(es) => {
			if let Ok(expanded_es) = es.expand() {
			    return Err(expanded_es.description().to_string());
			}
			else {
			    return Err("Internal Error".to_string());
			}
		    },
		    hdf5::Error::Internal(es) => {
			return Err(es);
		    }
		}
	    }
	};

	let hdf5_datasets = match f.datasets() {
	    Ok(ds) => ds,
	    Err(e) => {
		match e {
		    hdf5::Error::HDF5(es) => {
			if let Ok(expanded_es) = es.expand() {
			    return Err(expanded_es.description().to_string());
			}
			else {
			    return Err("Internal Error".to_string());
			}
		    },
		    hdf5::Error::Internal(es) => {
			return Err(es);
		    }
		}
	    }
	};

	for dataset in hdf5_datasets {
	    let name = dataset.name().clone();
	    println!("{}", dataset.name());
	    if let Ok(values) = dataset.read_raw::<T>() {
		let shape = dataset.shape();
		let t:Tensor<T> = Tensor::<T>::from_vector(shape.to_vec(), values);
		println!("{:?}", shape);
		datasets.insert(name.trim_start_matches('/').to_string(), t);
	    }
	}

	Ok(datasets)
    }
}
