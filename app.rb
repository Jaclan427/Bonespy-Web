
require 'sinatra'
require 'sinatra/reloader' if development?
require 'fileutils'
require 'open3'

# 🔥 REQUIRED FOR NGROK / TUNNELS (SINATRA 4)
set :bind, '0.0.0.0'

set :host_authorization, {
  permitted_hosts: [],
  allow_if: ->(env) { true }
}


# Set upload directory
UPLOADS_PATH = File.join(File.dirname(__FILE__), 'public', 'uploads')
FileUtils.mkdir_p(UPLOADS_PATH)

puts "=== SERVER STARTED ==="
puts "Uploads path: #{UPLOADS_PATH}"

# Home page: upload form
get '/' do
  erb :index
end

# Upload handler with YOLOv8 inference
post '/upload' do
  puts "=== UPLOAD DEBUG ==="
  
  if params[:file] && params[:file][:tempfile] && params[:file][:filename]
    tempfile = params[:file][:tempfile]
    filename = params[:file][:filename]
    
    # Sanitize filename
    safe_filename = filename.gsub(/[^\w\.\-]/, '_')
    input_path = File.join(UPLOADS_PATH, safe_filename)
    
    puts "Saving: #{filename} -> #{safe_filename}"
    
    # Write the uploaded file
    File.open(input_path, 'wb') do |f|
      f.write(tempfile.read)
    end
    
    if File.exist?(input_path)
      file_size = File.size(input_path)
      puts "SUCCESS: Original file saved! Size: #{file_size} bytes"
      
      # Run YOLOv8 inference
      annotated_filename = run_yolov8_inference(input_path, safe_filename)
      
      if annotated_filename
        @original_filename = safe_filename
        @annotated_filename = annotated_filename
        erb :upload
      else
        @message = "Upload successful but model inference failed."
        erb :index
      end
    else
      puts "ERROR: File not found after saving!"
      @message = "Upload failed - file not saved."
      erb :index
    end
  else
    puts "ERROR: No file in params"
    @message = "No file selected."
    erb :index
  end
end

def run_yolov8_inference(input_path, original_filename)
  begin
    # Generate output filename
    name_without_ext = File.basename(original_filename, '.*')
    output_filename = "#{name_without_ext}_annotated.jpg"
    
    puts "Running YOLOv8 inference on: #{input_path}"
    
    # YOLO command with low confidence threshold
    command = "yolo segment predict model=best.pt source=\"#{input_path}\" save=true project=\"public/uploads\" name=temp exist_ok=true conf=0.01"
    
    puts "Running command: #{command}"
    
    # Execute command
    stdout, stderr, status = Open3.capture3(command)
    
    puts "=== YOLO OUTPUT ==="
    puts "STDOUT: #{stdout}"
    puts "STDERR: #{stderr}" if stderr && !stderr.empty?
    puts "EXIT STATUS: #{status.exitstatus}"
    
    # Check if command was successful
    if status.success?
      puts "YOLO command executed successfully"
      
      # Look for the output in the temp directory
      temp_dir = "public/uploads/temp"
      if File.exist?(temp_dir)
        puts "Temp directory exists: #{temp_dir}"
        
        # Find all image files
        image_files = Dir.glob(File.join(temp_dir, "*.jpg")) +
                     Dir.glob(File.join(temp_dir, "*.png")) +
                     Dir.glob(File.join(temp_dir, "*.jpeg"))
        
        puts "Found #{image_files.size} image files: #{image_files}"
        
        if image_files.any?
          # Use the first image found
          source_file = image_files.first
          output_path = File.join(UPLOADS_PATH, output_filename)
          
          puts "Copying from #{source_file} to #{output_path}"
          FileUtils.cp(source_file, output_path)
          
          # Clean up temp directory
          FileUtils.rm_rf(temp_dir)
          
          if File.exist?(output_path)
            puts "SUCCESS: Annotated image saved to #{output_path}"
            return output_filename
          else
            puts "ERROR: Failed to copy file to #{output_path}"
          end
        else
          puts "ERROR: No image files found in temp directory"
        end
      else
        puts "ERROR: Temp directory not found: #{temp_dir}"
      end
    else
      puts "ERROR: YOLO command failed"
    end
    
    return nil
    
  rescue => e
    puts "Error during inference: #{e.message}"
    puts e.backtrace
    return nil
  end
end

# Serve uploaded images
get '/uploads/:filename' do |filename|
  file_path = File.join(UPLOADS_PATH, filename)
  
  puts "Image request: #{filename} -> #{file_path}"
  puts "File exists: #{File.exist?(file_path)}"
  
  if File.exist?(file_path)
    content_type 'image/jpeg' if filename.downcase.end_with?('.jpg', '.jpeg')
    content_type 'image/png' if filename.downcase.end_with?('.png')
    content_type 'image/gif' if filename.downcase.end_with?('.gif')
    
    send_file file_path
  else
    status 404
    "File not found: #{filename}"
  end
end

get '/Original.png' do
  send_file File.join(settings.public_folder, 'Original.png')
end

get '/templated.png' do
  send_file File.join(settings.public_folder, 'templated.png')
end


# Static file configuration
set :public_folder, File.join(File.dirname(__FILE__), 'public')
set :static, true

# Debug route
get '/debug' do
  files = Dir.glob(File.join(UPLOADS_PATH, '*'))
  content_type 'text/plain'
  "Uploads directory: #{UPLOADS_PATH}\n" +
  "Files:\n" + 
  files.map { |f| "  - #{File.basename(f)} (#{File.size(f)} bytes)" }.join("\n")
end