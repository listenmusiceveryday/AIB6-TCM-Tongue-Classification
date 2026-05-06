## Week 4
### recap ทำอะไรไปแล้วบ้าง
- ลองค้นหา healthy tongue dataset มาเพิ่มดูครับ เพราะ healthy tongue ที่มีอยู่แล้วมีเพียงแค่ <em>11 รูป</em> ในโฟลเดอร์ train และผมเจอ dataset ทั้งหมด 3 ตัว ผมคิดว่าจะดึงแค่ตัว healthy tongue มาเติมเต็ม

link01: https://link.springer.com/article/10.1186/s12880-024-01234-3

link02: https://pmc.ncbi.nlm.nih.gov/articles/PMC12025637

link03: https://pmc.ncbi.nlm.nih.gov/articles/PMC12921053

- ลองค้นหา Baseline ดูเรื่อยๆ ตอนนี้ยังไมได้ baseline ครับ (แต่ในความคิดอยากได้ Human Baseline ที่มาจากคุณหมอ) และผมลองหา Existing Models ดูระหว่างทางครับ<br>

<p>ซึ่งก็เจอตัวนึงเป็น <b>Convolutional Neural Networks in Tongue Lesions</b> มี 2 ตัว คือ <b>Binary Classification</b> ซึ่งความแม่นยำประมาณ 0.97 - 1.00 สูงมากครับ และ <b>Multi-class Classification</b> แยกแยะ 3 กลุ่ม ความแม่นยำประมาณ 0.49 - 0.99 แต่พวกนี้คือแยกพวกลักษณะพื้นผิวบนลิ้นครับ</p>

link01: https://pmc.ncbi.nlm.nih.gov/articles/PMC12921053

<p>และก็เจอ <b>Detecting Coated Tongue with Convolutional Neural Networks</b> เป็น Binary Classification แยกฝ้าครับ ความแม่นยำ 0.85 เป็นแค่แยก 2 กลุ่มเท่านั้นครับ คือ เป็นฝ้า vs ไม่เป็นฝ้า แต่ในส่วนของ สีของฝ้า เขาไม่ได้ทำครับ</p>

link01: https://pmc.ncbi.nlm.nih.gov/articles/PMC12025637

- ลองคิด metrics แล้ว คิดว่าจะใช้ Precision, Recall, F1-score และ Accuracy เป็น metric ครับ และก็ Confusion Matrices

- ลองไปศึกษาพวก Binary, Multi-label และ Multi-task ดูแล้ว และก็ไปเห็น paper ตัวนึงใช้ semantic segmentation นำมาคำนวน % พวกฝ้าบนลิ้นได้ (อันนี้เจ๋งมาก55555+) ตอนนี้ก็พอจะเห็นภาพในหัวแล้วว่าอยากจะให้ Project นี้ไปทางไหน
