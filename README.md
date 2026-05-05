### Week 4
#### recap ทำอะไรไปแล้วบ้าง
- ลองค้นหา dataset healthy มาเพิ่มดูครับ เพราะที่มีอยู่แล้ว มี healthy เพียงแค่ 11 รูป ใน train และผมเจอ dataset ทั้งหมด 3 ตัว ผมคิดว่าจะดึงแค่ตัว healthy มาเติมเต็ม

https://link.springer.com/article/10.1186/s12880-024-01234-3 

https://pmc.ncbi.nlm.nih.gov/articles/PMC12025637

https://pmc.ncbi.nlm.nih.gov/articles/PMC12921053

- ลองค้นหา Baseline ดูเรื่อยๆ ตอนนี้ยังไมได้ baseline ครับ (แต่ในความคิดอยากได้ Human Baseline ที่มาจากคุณหมอ) และผมลองหา Existing Models ดูระหว่างทางครับ<br>

<p>ซึ่งก็เจอตัวนึงเป็น Convolutional Neural Networks in Tongue Lesions มี 2 ตัว คือ Binary Classification ซึ่งความแม่นยำประมาณ 0.97 - 1.00 สูงมากครับ และ Multi-class Classification แยกแยะ 3 กลุ่ม ความแม่นยำประมาณ 0.49 - 0.99 แต่พวกนี้คือแยกพวกลักษณะพื้นผิวบนลิ้นครับ</p>

https://pmc.ncbi.nlm.nih.gov/articles/PMC12921053

<p>และก็เจอ Detecting Coated Tongue with Convolutional Neural Networks เป็น Binary Classification แยกฝ้าครับ ความแม่นยำ 0.85 เป็นแค่แยก 2 กลุ่มเท่านั้นครับ คือ เป็นฝ้า vs ไม่เป็นฝ้า แต่ในส่วนของ สีของฝ้า เขาไม่ได้ทำครับ</p>

https://pmc.ncbi.nlm.nih.gov/articles/PMC12025637 

- ลองคิด metrics แล้ว คิดว่าจะใช้ Precision, Recall, F1-score และ Accuracy เป็น metric ครับ และก็ Confusion Matrices
