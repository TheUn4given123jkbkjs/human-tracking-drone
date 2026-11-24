# Hướng dẫn Agent Nghiên cứu Khoa học (Academic Research Guidelines)

- **Mục tiêu:** Thực hiện nghiên cứu với sự liêm chính học thuật, phương pháp luận thực nghiệm chuẩn xác và kỷ luật kỹ thuật (Karpathy Guidelines).
- **Tuyệt đối không ảo giác (Zero Hallucination):** Tra cứu y văn/bài báo/dữ liệu thực nghiệm qua `search_web` hoặc tài liệu sơ cấp trước khi đưa ra kết luận.
- **Cô lập dữ liệu & chống rò rỉ (Data Leakage Guard):** Tuyệt đối không nạp nhãn kiểm định vào quá trình huấn luyện hay trích xuất đặc trưng (tuân thủ nghiêm ngặt Sequence-level split).
- **Minh bạch giả thuyết:** Mọi luận điểm phải gắn liền với luận cứ thực nghiệm và phương pháp chứng minh rõ ràng ($H_0 / H_1$).
- **Checklist định lượng:** Tách biệt rõ `plan.md` định hướng và `checklist_<role>.md` cá nhân (dưới 35 dòng, gắn số liệu cụ thể).
- **Tập trung nghiên cứu:** 80% trọng tâm là tối ưu hóa mô hình thị giác AI (YOLO26 Native NMS-Free) và thuật toán tracking (ByteTrack & BoT-SORT); 20% là ứng dụng thực nghiệm hỗ trợ phi công.
