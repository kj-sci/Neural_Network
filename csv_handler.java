import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class csv_handler {
	File fp;
	BufferedReader br;

	String indelim;
	String outdelim;

	String line;
	String[] data;
	String[] field_name;
	String[] data_type;
	
	/**************************************************************************************
	 *                                                                                    *
	 *                                                                                    *
	 *                    Constructor                                                     *
	 *                                                                                    *
	 *                                                                                    *
	 **************************************************************************************/

	public csv_handler(String d) {
		init_vars(d);
	}
	
	public csv_handler(String fname, String d) {
		init_vars(d);
		open_file(fname);
	}
	
	public void init_vars(String d) {
		indelim = d;
	}
	
	/**************************************************************************************
	 *                                                                                    *
	 *                                                                                    *
	 *                    File Handling                                                   *
	 *                                                                                    *
	 *                                                                                    *
	 **************************************************************************************/
	
	public int open_file (String fname) {
		try {
			fp = new File(fname);
			br = new BufferedReader(new FileReader(fp));
			return 0;
		} catch (Exception e) {
			return 1;
		}
	}
	
	public int close_file() {
		try {
			br.close();
			return 0;
		} catch (Exception e) {
			return 1;
		}
	}
	
	public boolean read_header() {
		if (read_line() == false) {
			return false;
		}
		field_name = data;
				
		return true;
	}

	public boolean read_line() {
		try {
			line = br.readLine();
			
			if (line == null) {
				return false;
			} else {
				data = line.split(indelim, -1);
				return true;
			}
		} catch (Exception e) {
			System.err.println("Error in csv_handler.read_line");
			return false;
		}
	}

	/**************************************************************************************
	 *                                                                                    *
	 *                                                                                    *
	 *                    Get Functions                                                   *
	 *                                                                                    *
	 *                                                                                    *
	 **************************************************************************************/
	public String[] get_data(){
		return data;
	}
	
	public int get_data_size() {
		return data.length;
	}
	
	public String[] get_field_name(){
		return field_name;
	}
	
	public String[] get_data_type(){
		return data_type;
	}

	
	/**************************************************************************************
	 *                                                                                    *
	 *                                                                                    *
	 *                    Print Functions                                                 *
	 *                                                                                    *
	 *                                                                                    *
	 **************************************************************************************/

	public void print_field_name(){
		System.out.println("----------------------- Field_Name ---------------------------------------");
		print_array(field_name);
	}

	public void print_data_type(){
		System.out.println("----------------------- Data_Type ---------------------------------------");
		print_array(data_type);
	}

	public void print_data(){
		System.out.println("----------------------- Data ---------------------------------------");
		print_array(data);
	}

	public void print_array(String[] array){
		System.out.println("-----------------------------------------------");
		for(int loop=0; loop<array.length; loop++) {
			System.out.println(loop+": "+array[loop]);
		}
		System.out.println("-----------------------------------------------");
	}


}
