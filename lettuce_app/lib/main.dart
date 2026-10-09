import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Lettuce Disease Detector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const DetectionScreen(),
    );
  }
}

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({super.key});

  @override
  State<DetectionScreen> createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  File? _selectedImage;
  bool _isLoading = false;
  String _resultText = 'กรุณาถ่ายรูปหรือเลือกรูปภาพผักกาดหอม';
  
  // URL สำหรับรันบน Web/Chrome
  final String _backendUrl =  'https://github.com/48580/lettuce-backend'
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(source: source);
    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
        _resultText = 'กดปุ่ม "วิเคราะห์โรคผัก" เพื่อส่งข้อมูล';
      });
    }
  }

  Future<Position?> _getCurrentLocation() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return null;

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) return null;
    }

    return await Geolocator.getCurrentPosition();
  }

  Future<void> _analyzeLettuce() async {
    if (_selectedImage == null) return;

    setState(() {
      _isLoading = true;
      _resultText = 'กำลังดึงพิกัด GPS และวิเคราะห์รูปภาพ...';
    });

    try {
      Position? position = await _getCurrentLocation();
      double lat = position?.latitude ?? 0.0;
      double lng = position?.longitude ?? 0.0;

      var request = http.MultipartRequest('POST', Uri.parse(_backendUrl));
      request.fields['lat'] = lat.toString();
      request.fields['lng'] = lng.toString();
      request.files.add(
        await http.MultipartFile.fromPath('file', _selectedImage!.path),
      );

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        setState(() {
          _resultText = 'ผลการตรวจ: ${data['disease_name'] ?? 'ไม่พบโรค'}\n'
              'โซนแปลง (3x3m): ${data['zone_id'] ?? 'N/A'}\n'
              'พิกัด: $lat, $lng';
        });
      } else {
        setState(() {
          _resultText = 'เกิดข้อผิดพลาดจากเซิร์ฟเวอร์ (${response.statusCode})';
        });
      }
    } catch (e) {
      setState(() {
        _resultText = 'ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้: $e';
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
        title: const Text('ระบบตรวจจับโรคผักกาดหอม'),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Container(
              height: 250,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.grey),
              ),
              child: _selectedImage != null
                  ? Image.file(_selectedImage!, fit: BoxFit.cover)
                  : const Center(
                      child: Icon(Icons.image, size: 80, color: Colors.grey),
                    ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('ถ่ายรูป'),
                ),
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('เลือกรูป'),
                ),
              ],
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isLoading || _selectedImage == null ? null : _analyzeLettuce,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                foregroundColor: Colors.white,
                minimumSize: const Size.fromHeight(45),
              ),
              child: _isLoading
                  ? const CircularProgressIndicator(color: Colors.white)
                  : const Text('วิเคราะห์โรคผัก', style: TextStyle(fontSize: 16)),
            ),
            const SizedBox(height: 20),
            Card(
              elevation: 2,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Text(
                  _resultText,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 15),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}