OPENQASM 2.0;
include "qelib1.inc";

qreg q[10];
creg rec[37];

reset q[0];
reset q[2];
reset q[3];
reset q[4];
reset q[5];
reset q[8];
reset q[9];
reset q[1];
reset q[6];
reset q[7];
barrier q;

reset q[6]; h q[6]; // decomposed RX
reset q[1]; h q[1]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
barrier q;

cx q[1], q[0];
cx q[7], q[4];
barrier q;

cx q[1], q[2];
cx q[7], q[8];
barrier q;

cx q[6], q[2];
cx q[7], q[3];
barrier q;

cx q[6], q[9];
cx q[1], q[4];
barrier q;

cx q[6], q[4];
cx q[7], q[5];
barrier q;

cx q[6], q[8];
cx q[1], q[3];
barrier q;

h q[6]; measure q[6] -> rec[0]; h q[6]; // decomposed MX
h q[1]; measure q[1] -> rec[1]; h q[1]; // decomposed MX
h q[7]; measure q[7] -> rec[2]; h q[7]; // decomposed MX
reset q[6];
reset q[1];
reset q[7];
barrier q;

cx q[0], q[1];
cx q[4], q[7];
barrier q;

cx q[2], q[1];
cx q[8], q[7];
barrier q;

cx q[2], q[6];
cx q[3], q[7];
barrier q;

cx q[9], q[6];
cx q[4], q[1];
barrier q;

cx q[4], q[6];
cx q[5], q[7];
barrier q;

cx q[8], q[6];
cx q[3], q[1];
barrier q;

measure q[6] -> rec[3];
measure q[1] -> rec[4];
measure q[7] -> rec[5];
barrier q;

reset q[6]; h q[6]; // decomposed RX
reset q[1]; h q[1]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
barrier q;

cx q[1], q[0];
cx q[7], q[4];
barrier q;

cx q[1], q[2];
cx q[7], q[8];
barrier q;

cx q[6], q[2];
cx q[7], q[3];
barrier q;

cx q[6], q[9];
cx q[1], q[4];
barrier q;

cx q[6], q[4];
cx q[7], q[5];
barrier q;

cx q[6], q[8];
cx q[1], q[3];
barrier q;

h q[6]; measure q[6] -> rec[6]; h q[6]; // decomposed MX
h q[1]; measure q[1] -> rec[7]; h q[1]; // decomposed MX
h q[7]; measure q[7] -> rec[8]; h q[7]; // decomposed MX
barrier q;

reset q[6];
reset q[1];
reset q[7];
barrier q;

cx q[0], q[1];
cx q[4], q[7];
barrier q;

cx q[2], q[1];
cx q[8], q[7];
barrier q;

cx q[2], q[6];
cx q[3], q[7];
barrier q;

cx q[9], q[6];
cx q[4], q[1];
barrier q;

cx q[4], q[6];
cx q[5], q[7];
barrier q;

cx q[8], q[6];
cx q[3], q[1];
barrier q;

measure q[6] -> rec[9];
measure q[1] -> rec[10];
measure q[7] -> rec[11];
barrier q;

reset q[6]; h q[6]; // decomposed RX
reset q[1]; h q[1]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
barrier q;

cx q[1], q[0];
cx q[7], q[4];
barrier q;

cx q[1], q[2];
cx q[7], q[8];
barrier q;

cx q[6], q[2];
cx q[7], q[3];
barrier q;

cx q[6], q[9];
cx q[1], q[4];
barrier q;

cx q[6], q[4];
cx q[7], q[5];
barrier q;

cx q[6], q[8];
cx q[1], q[3];
barrier q;

h q[6]; measure q[6] -> rec[12]; h q[6]; // decomposed MX
h q[1]; measure q[1] -> rec[13]; h q[1]; // decomposed MX
h q[7]; measure q[7] -> rec[14]; h q[7]; // decomposed MX
barrier q;

reset q[6];
reset q[1];
reset q[7];
barrier q;

cx q[0], q[1];
cx q[4], q[7];
barrier q;

cx q[2], q[1];
cx q[8], q[7];
barrier q;

cx q[2], q[6];
cx q[3], q[7];
barrier q;

cx q[9], q[6];
cx q[4], q[1];
barrier q;

cx q[4], q[6];
cx q[5], q[7];
barrier q;

cx q[8], q[6];
cx q[3], q[1];
barrier q;

measure q[6] -> rec[15];
measure q[1] -> rec[16];
measure q[7] -> rec[17];
barrier q;

reset q[6]; h q[6]; // decomposed RX
reset q[1]; h q[1]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
barrier q;

cx q[1], q[0];
cx q[7], q[4];
barrier q;

cx q[1], q[2];
cx q[7], q[8];
barrier q;

cx q[6], q[2];
cx q[7], q[3];
barrier q;

cx q[6], q[9];
cx q[1], q[4];
barrier q;

cx q[6], q[4];
cx q[7], q[5];
barrier q;

cx q[6], q[8];
cx q[1], q[3];
barrier q;

h q[6]; measure q[6] -> rec[18]; h q[6]; // decomposed MX
h q[1]; measure q[1] -> rec[19]; h q[1]; // decomposed MX
h q[7]; measure q[7] -> rec[20]; h q[7]; // decomposed MX
barrier q;

reset q[6];
reset q[1];
reset q[7];
barrier q;

cx q[0], q[1];
cx q[4], q[7];
barrier q;

cx q[2], q[1];
cx q[8], q[7];
barrier q;

cx q[2], q[6];
cx q[3], q[7];
barrier q;

cx q[9], q[6];
cx q[4], q[1];
barrier q;

cx q[4], q[6];
cx q[5], q[7];
barrier q;

cx q[8], q[6];
cx q[3], q[1];
barrier q;

measure q[6] -> rec[21];
measure q[1] -> rec[22];
measure q[7] -> rec[23];
barrier q;

reset q[6]; h q[6]; // decomposed RX
reset q[1]; h q[1]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
barrier q;

cx q[1], q[0];
cx q[7], q[4];
barrier q;

cx q[1], q[2];
cx q[7], q[8];
barrier q;

cx q[6], q[2];
cx q[7], q[3];
barrier q;

cx q[6], q[9];
cx q[1], q[4];
barrier q;

cx q[6], q[4];
cx q[7], q[5];
barrier q;

cx q[6], q[8];
cx q[1], q[3];
barrier q;

h q[6]; measure q[6] -> rec[24]; h q[6]; // decomposed MX
h q[1]; measure q[1] -> rec[25]; h q[1]; // decomposed MX
h q[7]; measure q[7] -> rec[26]; h q[7]; // decomposed MX
barrier q;

reset q[6];
reset q[1];
reset q[7];
barrier q;

cx q[0], q[1];
cx q[4], q[7];
barrier q;

cx q[2], q[1];
cx q[8], q[7];
barrier q;

cx q[2], q[6];
cx q[3], q[7];
barrier q;

cx q[9], q[6];
cx q[4], q[1];
barrier q;

cx q[4], q[6];
cx q[5], q[7];
barrier q;

cx q[8], q[6];
cx q[3], q[1];
barrier q;

measure q[6] -> rec[27];
measure q[1] -> rec[28];
measure q[7] -> rec[29];
barrier q;

measure q[0] -> rec[30];
measure q[2] -> rec[31];
measure q[3] -> rec[32];
measure q[4] -> rec[33];
measure q[5] -> rec[34];
measure q[8] -> rec[35];
measure q[9] -> rec[36];
