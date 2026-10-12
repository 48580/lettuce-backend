import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Lettuce Disease Detection',
      theme: ThemeData(
        primarySwatch: Colors.green,
        useMaterial3: true,
      ),
      home: const DetectionScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  Uint8List? _selectedImageBytes;
  XFile? _pickedFile;
  bool _isLoading = false;
  String _resultText = 'Please take a photo or select a lettuce image.';

  // Backend API URL on Render
  final String _backendUrl = 'https://lettuce-backend.onrender.com/predict';
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? image = await _picker.pickImage(
        source: source,
        imageQuality: 80,
      );
      if (image != null) {
        final bytes = await image.readAsBytes();
        setState(() {
          _pickedFile = image;
          _selectedImageBytes = bytes;
          _resultText = 'Click "Analyze Disease" to process.';
        });
      }
    } catch (e) {
      setState(() {
        _resultText = 'Error selecting image: $e';
      });
    }
  }

  Future<void> _uploadImage() async {
    if (_selectedImageBytes == null || _pickedFile == null) {
      setState(() {
        _resultText = 'Please select an image first.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _resultText = 'Analyzing image... (May take a few seconds)';
    });

    try {
      var request = http.MultipartRequest('POST', Uri.parse(_backendUrl));
      
      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          _selectedImageBytes!,
          filename: _pickedFile!.name,
        ),
      );

      request.fields['lat'] = '0.0';
      request.fields['lng'] = '0.0';

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        try {
          final decodedBody = jsonDecode(response.body);
          
          // ตรวจสอบรูปแบบข้อมูลที่เซิร์ฟเวอร์ส่งกลับมาอย่างปลอดภัย
          if (decodedBody is Map<String, dynamic>) {
            String diseaseClass = decodedBody['class']?.toString() ?? decodedBody['error']?.toString() ?? 'Unknown';
            var confidence = decodedBody['confidence'] ?? 0;
            
            if (confidence is double) {
              confidence = (confidence * 100).toStringAsFixed(2);
            }

            setState(() {
              _resultText = 'Analysis Result: $diseaseClass\nConfidence: $confidence%';
            });
          } else if (decodedBody is List) {
            setState(() {
              _resultText = 'Server returned a list: ${decodedBody.toString()}';
            });
          } else {
            setState(() {
              _resultText = 'Server response: ${decodedBody.toString()}';
            });
          }
        } catch (e) {
          setState(() {
            _resultText = 'Failed to parse JSON: ${response.body}';
          });
        }
      } else {
        setState(() {
          _resultText = 'Analysis failed (${response.statusCode}):\n${response.body}';
        });
      }
    } catch (e) {
      setState(() {
        _resultText = 'Failed to connect to server: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Lettuce Disease Detection System'),
        centerTitle: true,
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              height: 300,
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.grey.shade400),
              ),
              child: _selectedImageBytes != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.memory(
                        _selectedImageBytes!,
                        fit: BoxFit.contain,
                      ),
                    )
                  : const Center(
                      child: Text(
                        'No image selected',
                        style: TextStyle(color: Colors.grey, fontSize: 16),
                      ),
                    ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green.shade100,
                    foregroundColor: Colors.green.shade900,
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green.shade100,
                    foregroundColor: Colors.green.shade900,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isLoading ? null : _uploadImage,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 14),
                textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              child: _isLoading
                  ? const CircularProgressIndicator(color: Colors.white)
                  : const Text('Analyze Disease'),
            ),
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey.shade300),
              ),
              child: Text(
                _resultText,
                style: const TextStyle(fontSize: 16),
                textAlign: TextAlign.center,
              ),
            ),
          ],
        ),
      ),
    );
  }
}