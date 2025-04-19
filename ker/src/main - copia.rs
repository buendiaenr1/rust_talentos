use ndarray::{Array1, Array2};
use std::io::{self, Write};

use ndarray::Axis;

use std::fs::File;
use std::io::{BufReader, BufRead};
use std::process::Command;


fn read_csv_and_process(file_path: &str) -> Array2<f64> {
    // println!(" leo csv...");
    // Abrir el archivo CSV
    let file = File::open(file_path).expect("No se pudo abrir el archivo");
    let reader = BufReader::new(file);

    // Leer el archivo línea por línea
    let mut data: Vec<f64> = Vec::new();
    let mut rows = 0;
    let mut cols = 0;

    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("Error al leer la línea");
        let values: Vec<&str> = line.split(',').collect();

        // Determinar el número de columnas en la primera iteración
        if i == 0 {
            cols = values.len() - 1; // Restamos 1 porque eliminaremos la última columna
        }

        // Ignorar la última columna y convertir las demás a f64
        for value in &values[..cols] {
            data.push(value.parse::<f64>().expect("Error al parsear el valor"));
        }
        rows += 1;
        //println!(" linea {} ",rows);
    }

    println!(" renglones {} con {} columnas ...",rows,cols);
    // Crear la matriz Array2 con los datos procesados
    Array2::from_shape_vec((rows, cols), data).expect("Error al crear la matriz")
}

/// Función para calcular la matriz de distancia euclidiana
fn compute_distance_matrix(data: &Array2<f64>) -> Array2<f64> {
    let n = data.len_of(Axis(0));
    let mut distance_matrix = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = (&data.row(i) - &data.row(j)).mapv(|x| x.powi(2)).sum().sqrt();
            distance_matrix[[i, j]] = dist;
            distance_matrix[[j, i]] = dist; // Simetría
        }
    }

    distance_matrix
}

/// Función para realizar clustering jerárquico aglomerativo (HACA)
fn hierarchical_agglomerative_clustering(
    data: &Array2<f64>,
    num_clusters: usize,
) -> Vec<usize> {
    let n = data.len_of(Axis(0));
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let distance_matrix = compute_distance_matrix(data);

    while clusters.len() > num_clusters {
        // Encontrar el par de clusters con la menor distancia
        let mut min_dist = f64::MAX;
        let mut merge_indices = (0, 0);

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let mut dist = 0.0;
                for &idx1 in &clusters[i] {
                    for &idx2 in &clusters[j] {
                        dist += distance_matrix[[idx1, idx2]];
                    }
                }
                dist /= (clusters[i].len() * clusters[j].len()) as f64; // Promedio

                if dist < min_dist {
                    min_dist = dist;
                    merge_indices = (i, j);
                }
            }
        }

        // Fusionar los dos clusters más cercanos
        let (i, j) = merge_indices;
        if i < j {
            let (left, right) = clusters.split_at_mut(j);
            left[i].extend(&right[0]);
            clusters.remove(j);
        } else {
            let (left, right) = clusters.split_at_mut(i);
            right[0].extend(&left[j]);
            clusters.remove(i);
        }
    }

   

    // Asignar etiquetas de cluster a cada punto
    let mut labels = vec![0; n];
    for (cluster_id, cluster) in clusters.iter().enumerate() {
        for &point_idx in cluster {
            labels[point_idx] = cluster_id;
        }
    }

    labels
}


// Definir los kernels
fn linear_kernel(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    x.dot(y)
}

fn polynomial_kernel(x: &Array1<f64>, y: &Array1<f64>, degree: usize) -> f64 {
    (x.dot(y) + 1.0).powi(degree as i32)
}

fn rbf_kernel(x: &Array1<f64>, y: &Array1<f64>, gamma: f64) -> f64 {
    (-gamma * (x - y).mapv(|v| v.powi(2)).sum()).exp()
}

// Kernel sigmoidal
fn sigmoid_kernel(x: &Array1<f64>, y: &Array1<f64>, alpha: f64, c: f64) -> f64 {
    (alpha * x.dot(y) + c).tanh()
}
// Función específica para alpha=0.1 y c=0.0
fn sigmoid_kernel_fixed(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let alpha = 0.1;
    let c = 0.0;
    sigmoid_kernel(x, y, alpha, c)
}



// Función para calcular la matriz de kernel
fn compute_kernel_matrix(
    data: &Array2<f64>,
    kernel: fn(&Array1<f64>, &Array1<f64>) -> f64,
) -> Array2<f64> {
    let n = data.len_of(ndarray::Axis(0));
    let mut kernel_matrix = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            kernel_matrix[[i, j]] = kernel(&data.row(i).to_owned(), &data.row(j).to_owned());
        }
    }
    kernel_matrix
}

// Implementación simplificada del SVM
struct SVM {
    alpha: Array1<f64>,
    bias: f64,
    support_vectors: Array2<f64>,
    labels: Array1<f64>,
    kernel_fn: fn(&Array1<f64>, &Array1<f64>) -> f64,
}

impl SVM {
    fn new() -> Self {
        SVM {
            alpha: Array1::zeros(0),
            bias: 0.0,
            support_vectors: Array2::zeros((0, 0)),
            labels: Array1::zeros(0),
            kernel_fn: linear_kernel,
        }
    }

    fn fit(
        &mut self,
        data: &Array2<f64>,
        labels: &Array1<f64>,
        kernel_fn: fn(&Array1<f64>, &Array1<f64>) -> f64,
        c: f64,
    ) {
        let n = data.len_of(ndarray::Axis(0));
        self.alpha = Array1::zeros(n);
        self.bias = 0.0;
        self.support_vectors = data.clone();
        self.labels = labels.clone();
        self.kernel_fn = kernel_fn;

        let kernel_matrix = compute_kernel_matrix(data, kernel_fn);

        for _ in 0..100 { // Iteraciones (ajustar según sea necesario)
            for i in 0..n {
                let e_i = self.predict_one(&data.row(i).to_owned()) - labels[i];
                if (labels[i] * e_i < -0.01 && self.alpha[i] < c) || (labels[i] * e_i > 0.01 && self.alpha[i] > 0.0) {
                    let j = rand::random_range(0..n);
                    if i == j {
                        continue;
                    }

                    let _alpha_old_i = self.alpha[i];
                    let alpha_old_j = self.alpha[j];

                    let e_j = self.predict_one(&data.row(j).to_owned()) - labels[j];

                    let mut l = if labels[i] != labels[j] {
                        (self.alpha[j] - self.alpha[i]).max(0.0)
                    } else {
                        (self.alpha[j] + self.alpha[i] - c).max(0.0)
                    };
                    let mut h = if labels[i] != labels[j] {
                        (self.alpha[j] - self.alpha[i] + c).min(c)
                    } else {
                        (self.alpha[j] + self.alpha[i]).min(c)
                    };

                    // Verificar que l <= h
                    if l > h {l=0.0; h=0.0;}
                    if l > h {
                        println!("Error: l > h. l = {}, h = {}", l, h);
                        continue; // Saltar esta iteración si los límites son inválidos
                    }

                    if l == h {
                        continue;
                    }

                    let eta = 2.0 * kernel_matrix[[i, j]] - kernel_matrix[[i, i]] - kernel_matrix[[j, j]];
                    if eta >= 0.0 {
                        continue;
                    }

                    self.alpha[j] -= labels[j] * (e_i - e_j) / eta;
                    self.alpha[j] = if labels[i] != labels[j] {
                        self.alpha[j].clamp(l, h)
                    } else {
                        self.alpha[j].clamp(l, h)
                    };

                    if (self.alpha[j] - alpha_old_j).abs() < 1e-5 {
                        continue;
                    }

                    self.alpha[i] += labels[i] * labels[j] * (alpha_old_j - self.alpha[j]);
                }
            }
        }

        let mut b_sum = 0.0;
        let mut count = 0;
        for i in 0..n {
            if self.alpha[i] > 0.0 && self.alpha[i] < c {
                b_sum += labels[i] - self.predict_one(&data.row(i).to_owned());
                count += 1;
            }
        }
        self.bias = b_sum / count as f64;
    }

    fn predict_one(&self, x: &Array1<f64>) -> f64 {
        self.support_vectors
            .outer_iter()
            .zip(self.alpha.iter())
            .zip(self.labels.iter())
            .map(|((sv, &alpha), &label)| {
                alpha * label * (self.kernel_fn)(x, &sv.to_owned())
            })
            .sum::<f64>()
            + self.bias
    }

    fn predict(&self, data: &Array2<f64>) -> Array1<f64> {
        data.outer_iter()
            .map(|x| self.predict_one(&x.to_owned()))
            .map(|v| if v > 0.0 { 1.0 } else { -1.0 })
            .collect()
    }
}

// Función para calcular la matriz de confusión
fn confusion_matrix(predictions: &Array1<f64>, true_labels: &Array1<f64>) -> [[usize; 2]; 2] {
    let mut matrix = [[0; 2]; 2]; // [[TN, FP], [FN, TP]]
    println!(" [[TN, FP], [FN, TP]]");
    for (&pred, &true_label) in predictions.iter().zip(true_labels.iter()) {
        if pred == 1.0 && true_label == 1.0 {
            matrix[1][1] += 1; // Verdadero Positivo (TP)
        } else if pred == 1.0 && true_label == -1.0 {
            matrix[0][1] += 1; // Falso Positivo (FP)
        } else if pred == -1.0 && true_label == 1.0 {
            matrix[1][0] += 1; // Falso Negativo (FN)
        } else {
            matrix[0][0] += 1; // Verdadero Negativo (TN)
        }
    }
    matrix
}

/// Función para calcular ACC (Accuracy)
fn accuracy(matrix: [[usize; 2]; 2]) -> f64 {
    let tn = matrix[0][0];
    let fp = matrix[0][1];
    let fn_ = matrix[1][0];
    let tp = matrix[1][1];
    let total = tn + fp + fn_ + tp;
    if total == 0 {
        0.0
    } else {
        (tp + tn) as f64 / total as f64
    }
}

// Función para calcular SENS (Sensitivity o True Positive Rate)
fn sensitivity(matrix: [[usize; 2]; 2]) -> f64 {
    let fn_ = matrix[1][0];
    let tp = matrix[1][1];
    if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_) as f64
    }
}

// Función para calcular SPEC (Specificity o True Negative Rate)
fn specificity(matrix: [[usize; 2]; 2]) -> f64 {
    let tn = matrix[0][0];
    let fp = matrix[0][1];
    if tn + fp == 0 {
        0.0
    } else {
        tn as f64 / (tn + fp) as f64
    }
}

// Función para calcular PREC (Precision)
fn precision(matrix: [[usize; 2]; 2]) -> f64 {
    let fp = matrix[0][1];
    let tp = matrix[1][1];
    if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    }
}

// Función para calcular ERR (Error Rate)
fn error_rate(matrix: [[usize; 2]; 2]) -> f64 {
    let tn = matrix[0][0];
    let fp = matrix[0][1];
    let fn_ = matrix[1][0];
    let tp = matrix[1][1];
    let total = tn + fp + fn_ + tp;
    if total == 0 {
        0.0
    } else {
        (fp + fn_) as f64 / total as f64
    }
}

// Función para calcular MCC (Matthews Correlation Coefficient)
fn mcc(matrix: [[usize; 2]; 2]) -> f64 {
    let tn = matrix[0][0];
    let fp = matrix[0][1];
    let fn_ = matrix[1][0];
    let tp = matrix[1][1];
    let numerator = (tp * tn) as f64 - (fp * fn_) as f64;
    let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)) as f64;
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator.sqrt()
    }
}


fn main() {
    // limpiar
    Command::new("cmd")
            .args(&["/C", "cls"])
            .status()
            .expect("Error al ejecutar el comando cls");
        println!(" \n\n\n BUAP 2025: Enrique Buendia Lozada");
        println!(" Selección de talentos usando resultados de pruebas en dd.csv\n\n ");
    // HACA  *****************
    // Ejemplo de datos: matriz de características (4 arqueros, 2 variables)
    //let data = Array2::from_shape_vec(
    //    (4, 2),
    //    vec![
    //        5.1, 3.5, // Arquero 1
    //        4.9, 3.0, // Arquero 2
    //        6.2, 3.4, // Arquero 3
    //        5.0, 3.2, // Arquero 4
    //        
    //    ],
    //)
    //.unwrap();

    //println!(" ini...");
    // Ruta del archivo CSV
    let file_path = "dd.csv";

    // Procesar el archivo CSV
    let data = read_csv_and_process(file_path);

    // Imprimir la matriz resultante
    // println!("{:?}", data);


    //let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    //let labels = Array1::from(vec![-1.0, 1.0, 1.0, -1.0]);
    //let test_data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    let test_data = data.clone();

    // Realizar clustering jerárquico aglomerativo
    let num_clusters = 2; // Dividir en 2 grupos: HPA y LPA
    //let labels = hierarchical_agglomerative_clustering(&data, num_clusters);
    // Obtener las etiquetas de cluster como un vector de enteros
    let labels_vec = hierarchical_agglomerative_clustering(&data, num_clusters);
    // Convertir el vector de enteros a un Array1<f64>
    //let labels: Array1<f64> = Array1::from(labels_vec.iter().map(|&x| x as f64).collect::<Vec<f64>>());
    // Convertir las etiquetas del clustering (0 = LPA, 1 = HPA) a (-1 = LPA, 1 = HPA)
    let labels: Array1<f64> = Array1::from(labels_vec.iter().map(|&x| if x == 0 { -1.0 } else { 1.0 }).collect::<Vec<f64>>());



    // Mostrar los resultados del clustering
    println!("Asignación de clusters -1=L  1=H  :");
    //for (i, label) in labels.iter().enumerate() {
    //    println!("Arquero {}: Cluster {}", i + 1, label);
    //}
    // HACA ****************


    //let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    //let labels = Array1::from(vec![-1.0, 1.0, 1.0, -1.0]);
    //let test_data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();


    let mut svm = SVM::new();
    let mut best_kernel_fn: fn(&Array1<f64>, &Array1<f64>) -> f64 = linear_kernel;
    let mut best_accuracy = 0.0;
    let mut best_kernel_name = String::new(); // Cambiado a String

    
    // Kernel Lineal
    svm.fit(&data, &labels, linear_kernel, 1.0);
    println!(" ......");
    let predictions_linear = svm.predict(&test_data);
    //println!("Predicciones con kernel lineal: {:?}", predictions_linear);
    let matrix_linear = confusion_matrix(&predictions_linear, &labels);
    println!("Matriz de confusión (Kernel Lineal): {:?}", matrix_linear);
    
    // Calcular métricas
    let acc = accuracy(matrix_linear);
    let sens = sensitivity(matrix_linear);
    let spec = specificity(matrix_linear);
    let prec = precision(matrix_linear);
    let err = error_rate(matrix_linear);
    let mcc_value = mcc(matrix_linear);

    println!("Accuracy (ACC): {:.2}%", acc * 100.0);
    println!("Sensitivity (SENS): {:.2}%", sens * 100.0);
    println!("Specificity (SPEC): {:.2}%", spec * 100.0);
    println!("Precision (PREC): {:.2}%", prec * 100.0);
    println!("Error Rate (ERR): {:.2}%", err * 100.0);
    println!("Matthews Correlation Coefficient (MCC): {:.4}", mcc_value);
    println!("Precisión (Kernel Lineal): {:.2}%", acc * 100.0);
    if acc > best_accuracy {
        best_accuracy = acc;
        best_kernel_fn = linear_kernel;
        best_kernel_name = "Kernel Lineal".to_string(); // Asignación directa
    }

    // Kernel Polinomial (degree = 2)
    let c_values = vec![0.1, 1.0, 10.0];
    for c in c_values {
        let mut svm = SVM::new();
        fn poly_fixed_with_degree(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
            polynomial_kernel(x, y, 2)
        }
        svm.fit(&data, &labels, poly_fixed_with_degree, c);
        let predictions_poly = svm.predict(&test_data);
        //println!("\nPredicciones con kernel polinomial 2 (C = {}): {:?}", c, predictions_poly);
        let matrix_poly = confusion_matrix(&predictions_poly, &labels);
        println!("Matriz de confusión (Kernel Polinomial): {:?}", matrix_poly);
        // Calcular métricas
        let acc = accuracy(matrix_poly);
        let sens = sensitivity(matrix_poly);
        let spec = specificity(matrix_poly);
        let prec = precision(matrix_poly);
        let err = error_rate(matrix_poly);
        let mcc_value = mcc(matrix_poly);

        println!("Accuracy (ACC): {:.2}%", acc * 100.0);
        println!("Sensitivity (SENS): {:.2}%", sens * 100.0);
        println!("Specificity (SPEC): {:.2}%", spec * 100.0);
        println!("Precision (PREC): {:.2}%", prec * 100.0);
        println!("Error Rate (ERR): {:.2}%", err * 100.0);
        println!("Matthews Correlation Coefficient (MCC): {:.4}", mcc_value);
        println!("Precisión (Kernel Polinomial 2) con C= {} es: {:.2}%", c, acc * 100.0);
        if acc > best_accuracy {
            best_accuracy = acc;
            best_kernel_fn = poly_fixed_with_degree;
            best_kernel_name = format!("Kernel Polinomial 2 con C = {}", c); // Asignación directa
        }
    }
    // Kernel Polinomial (degree = 3)
    let c_values = vec![0.1, 1.0, 10.0];
    for c in c_values {
        let mut svm = SVM::new();
        fn poly_fixed_with_degree(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
            polynomial_kernel(x, y, 3)
        }
        svm.fit(&data, &labels, poly_fixed_with_degree, c);
        let predictions_poly = svm.predict(&test_data);
        //println!("\nPredicciones con kernel polinomial 3 (C = {}): {:?}", c, predictions_poly);
        let matrix_poly = confusion_matrix(&predictions_poly, &labels);
        println!("Matriz de confusión (Kernel Polinomial): {:?}", matrix_poly);
        // Calcular métricas
        let acc = accuracy(matrix_poly);
        let sens = sensitivity(matrix_poly);
        let spec = specificity(matrix_poly);
        let prec = precision(matrix_poly);
        let err = error_rate(matrix_poly);
        let mcc_value = mcc(matrix_poly);

        println!("Accuracy (ACC): {:.2}%", acc * 100.0);
        println!("Sensitivity (SENS): {:.2}%", sens * 100.0);
        println!("Specificity (SPEC): {:.2}%", spec * 100.0);
        println!("Precision (PREC): {:.2}%", prec * 100.0);
        println!("Error Rate (ERR): {:.2}%", err * 100.0);
        println!("Matthews Correlation Coefficient (MCC): {:.4}", mcc_value);
        println!("Precisión (Kernel Polinomial 3) con C= {} es: {:.2}%", c, acc * 100.0);
        if acc > best_accuracy {
            best_accuracy = acc;
            best_kernel_fn = poly_fixed_with_degree;
            best_kernel_name = format!("Kernel Polinomial 3 con C = {}", c); // Asignación directa
        }
    }

    // Kernel RBF (gamma = 1.0)
    const GAMMA: f64 = 1.0;
    fn rbf_fixed_with_gamma(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        rbf_kernel(x, y, GAMMA)
    }
    svm.fit(&data, &labels, rbf_fixed_with_gamma, 1.0);
    let predictions_rbf = svm.predict(&test_data);
    //println!("\nPredicciones con kernel RBF: {:?}", predictions_rbf);
    let matrix_rbf = confusion_matrix(&predictions_rbf, &labels);
    println!("Matriz de confusión (Kernel RBF): {:?}", matrix_rbf);
    // Calcular métricas
    let acc = accuracy(matrix_rbf);
    let sens = sensitivity(matrix_rbf);
    let spec = specificity(matrix_rbf);
    let prec = precision(matrix_rbf);
    let err = error_rate(matrix_rbf);
    let mcc_value = mcc(matrix_rbf);

    println!("Accuracy (ACC): {:.2}%", acc * 100.0);
    println!("Sensitivity (SENS): {:.2}%", sens * 100.0);
    println!("Specificity (SPEC): {:.2}%", spec * 100.0);
    println!("Precision (PREC): {:.2}%", prec * 100.0);
    println!("Error Rate (ERR): {:.2}%", err * 100.0);
    println!("Matthews Correlation Coefficient (MCC): {:.4}", mcc_value);
    println!("Precisión (Kernel RBF): {:.2}%", acc * 100.0);
    if acc > best_accuracy {
        best_accuracy = acc;
        best_kernel_fn = rbf_fixed_with_gamma;
        best_kernel_name = "Kernel RBF".to_string(); // Asignación directa
    }

    // Entrenamiento con kernel sigmoidal
    svm.fit(&data, &labels, sigmoid_kernel_fixed, 1.0);
    let predictions_sigmoid = svm.predict(&test_data);
    //println!("\nPredicciones con kernel sigmoidal: {:?}", predictions_sigmoid);
    let matrix_sigmoid = confusion_matrix(&predictions_sigmoid, &labels);
    println!("Matriz de confusión (Kernel Sigmoidal): {:?}", matrix_sigmoid);
    // Calcular métricas
    let acc = accuracy(matrix_sigmoid);
    let sens = sensitivity(matrix_sigmoid);
    let spec = specificity(matrix_sigmoid);
    let prec = precision(matrix_sigmoid);
    let err = error_rate(matrix_sigmoid);
    let mcc_value = mcc(matrix_sigmoid);

    println!("Accuracy (ACC): {:.2}%", acc * 100.0);
    println!("Sensitivity (SENS): {:.2}%", sens * 100.0);
    println!("Specificity (SPEC): {:.2}%", spec * 100.0);
    println!("Precision (PREC): {:.2}%", prec * 100.0);
    println!("Error Rate (ERR): {:.2}%", err * 100.0);
    println!("Matthews Correlation Coefficient (MCC): {:.4}", mcc_value);
    println!("Precisión (Kernel Sigmoidal) con alpha=0.1 y c=1.0: {:.2}%", acc * 100.0);
    if acc > best_accuracy {
        best_accuracy = acc;
        best_kernel_fn = sigmoid_kernel_fixed;
        best_kernel_name = "Kernel Sigmoidal con alpha=0.1 y c=1.0".to_string(); // Asignación directa
    }

    // Mostrar el mejor kernel
    println!(
        "\nEl mejor kernel es '{}' con una precisión global de {:.2}%.",
        best_kernel_name,
        best_accuracy * 100.0
    );

    // Leer nuevos datos desde la entrada estándar
    println!("\nIngrese nuevos datos para predecir con el mejor kernel '{}':", best_kernel_name);
    println!("Formato: x1,x2,...,xn (ejemplo: 0.5,0.3,...). Escriba 'salir' para terminar.");
    println!("El número de características debe ser {}.", data.len_of(Axis(1)));

    let mut input = String::new();
    loop {
        print!("Nuevo dato: ");
        io::stdout().flush().unwrap(); // Forzar impresión del mensaje
        input.clear();
        io::stdin().read_line(&mut input).expect("Error al leer entrada");

        let input = input.trim();
        if input == "salir" {
            break;
        }

        // Parsear los datos ingresados
        let parts: Vec<&str> = input.split(',').collect();
        if parts.len() != data.len_of(Axis(1)) {
            println!(
                "Entrada inválida. Debe ingresar exactamente {} valores separados por comas.",
                data.len_of(Axis(1))
            );
            continue;
        }

        // Convertir los valores a f64
        let new_data: Result<Vec<f64>, _> = parts.iter().map(|&val| val.parse::<f64>()).collect();
        let new_data = match new_data {
            Ok(vec) => vec,
            Err(_) => {
                println!("Uno o más valores no son números válidos. Intente de nuevo.");
                continue;
            }
        };

        // Crear un nuevo vector de datos
        let new_data = Array1::from(new_data);

        // Predecir usando el mejor kernel
        let prediction = svm.predict_one_with_kernel(&new_data, best_kernel_fn);
        println!(
            "Predicción para {:?}:---> {}",
            new_data,
            if prediction > 0.0 { 1.0 } else { -1.0 }
        );
    }
    // Mantener la consola abierta después de salir
    Command::new("cmd")
            .args(&["/C", "cmd /k"])
            .status()
            .expect("Error al ejecutar el comando cmd /k");

}

// Extensión del struct SVM para permitir predicciones con un kernel específico
impl SVM {
    fn predict_one_with_kernel(&self, x: &Array1<f64>, kernel_fn: fn(&Array1<f64>, &Array1<f64>) -> f64) -> f64 {
        self.support_vectors
            .outer_iter()
            .zip(self.alpha.iter())
            .zip(self.labels.iter())
            .map(|((sv, &alpha), &label)| {
                alpha * label * kernel_fn(x, &sv.to_owned())
            })
            .sum::<f64>()
            + self.bias
    }
}