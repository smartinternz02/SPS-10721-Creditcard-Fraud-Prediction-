package fp.ml;


import java.io.IOException;

//import javax.sql.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Linear {
		public static void main(String[] args) throws Exception {
			
			DataSource source =new DataSource("C:\\Users\\anjumfirdous\\eclipse-workspace\\fp.ml\\src\\main\\java\\fp\\ml\\creditcardfinalmod.csv");
			Instances dataset=source.getDataSet();
			dataset.setClassIndex(dataset.numAttributes()-1);
			//linear Regression
			LinearRegression lr=new LinearRegression();
			lr.buildClassifier(dataset);
			
			Evaluation lreval =new Evaluation(dataset);
		    lreval.evaluateModel(lr,dataset);
			System.out.println(lreval.toSummaryString());
			
			
		}

	}

