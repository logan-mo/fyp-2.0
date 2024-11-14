create table car_data(
	car_id VARCHAR primary key,
	car_type text,
	car_color text,
	car_licence_plate_color text,
	car_license_plate_number text
);

create table camera_data(
	camera_id VARCHAR primary key,
	location_x DECIMAL not NULL,
	location_y DECIMAL not NULL
);

create table detections(

	detection_id INT primary key,
	camera_id VARCHAR,
	car_type VARCHAR,
	car_color VARCHAR,
	car_licence_plate_number VARCHAR,
	datetime timestamp,
	
	-- foreign key (camera_id) references camera_data(camera_id),
	-- foreign key (car_id) references car_data(car_id)
);