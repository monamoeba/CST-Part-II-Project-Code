OPENQASM 2.0;
include "qelib1.inc";

qreg q[10];
creg rec[37];

reset q[0];
reset q[1];
reset q[2];
reset q[3];
reset q[4];
reset q[5];
reset q[6];
reset q[7];
reset q[8];
reset q[9];
barrier q;

reset q[8];
reset q[7];
reset q[9];
h q[8];
h q[7];
h q[9];
barrier q;

cx q[7], q[0];
cx q[9], q[3];
barrier q;

cx q[7], q[1];
cx q[9], q[5];
barrier q;

cx q[8], q[1];
cx q[9], q[2];
barrier q;

cx q[8], q[6];
cx q[7], q[3];
barrier q;

cx q[8], q[3];
cx q[9], q[4];
barrier q;

cx q[8], q[5];
cx q[7], q[2];
barrier q;

h q[8];
h q[7];
h q[9];
measure q[8] -> rec[0];
measure q[7] -> rec[1];
measure q[9] -> rec[2];
h q[8];
h q[7];
h q[9];
reset q[8];
reset q[7];
reset q[9];
barrier q;

cx q[0], q[7];
cx q[3], q[9];
barrier q;

cx q[1], q[7];
cx q[5], q[9];
barrier q;

cx q[1], q[8];
cx q[2], q[9];
barrier q;

cx q[6], q[8];
cx q[3], q[7];
barrier q;

cx q[3], q[8];
cx q[4], q[9];
barrier q;

cx q[5], q[8];
cx q[2], q[7];
barrier q;

measure q[8] -> rec[3];
measure q[7] -> rec[4];
measure q[9] -> rec[5];
barrier q;

reset q[8];
reset q[7];
reset q[9];
h q[8];
h q[7];
h q[9];
barrier q;

cx q[7], q[0];
cx q[9], q[3];
barrier q;

cx q[7], q[1];
cx q[9], q[5];
barrier q;

cx q[8], q[1];
cx q[9], q[2];
barrier q;

cx q[8], q[6];
cx q[7], q[3];
barrier q;

cx q[8], q[3];
cx q[9], q[4];
barrier q;

cx q[8], q[5];
cx q[7], q[2];
barrier q;

h q[8];
h q[7];
h q[9];
measure q[8] -> rec[6];
measure q[7] -> rec[7];
measure q[9] -> rec[8];
h q[8];
h q[7];
h q[9];
barrier q;

reset q[8];
reset q[7];
reset q[9];
barrier q;

cx q[0], q[7];
cx q[3], q[9];
barrier q;

cx q[1], q[7];
cx q[5], q[9];
barrier q;

cx q[1], q[8];
cx q[2], q[9];
barrier q;

cx q[6], q[8];
cx q[3], q[7];
barrier q;

cx q[3], q[8];
cx q[4], q[9];
barrier q;

cx q[5], q[8];
cx q[2], q[7];
barrier q;

measure q[8] -> rec[9];
measure q[7] -> rec[10];
measure q[9] -> rec[11];
barrier q;

reset q[8];
reset q[7];
reset q[9];
h q[8];
h q[7];
h q[9];
barrier q;

cx q[7], q[0];
cx q[9], q[3];
barrier q;

cx q[7], q[1];
cx q[9], q[5];
barrier q;

cx q[8], q[1];
cx q[9], q[2];
barrier q;

cx q[8], q[6];
cx q[7], q[3];
barrier q;

cx q[8], q[3];
cx q[9], q[4];
barrier q;

cx q[8], q[5];
cx q[7], q[2];
barrier q;

h q[8];
h q[7];
h q[9];
measure q[8] -> rec[12];
measure q[7] -> rec[13];
measure q[9] -> rec[14];
h q[8];
h q[7];
h q[9];
barrier q;

reset q[8];
reset q[7];
reset q[9];
barrier q;

cx q[0], q[7];
cx q[3], q[9];
barrier q;

cx q[1], q[7];
cx q[5], q[9];
barrier q;

cx q[1], q[8];
cx q[2], q[9];
barrier q;

cx q[6], q[8];
cx q[3], q[7];
barrier q;

cx q[3], q[8];
cx q[4], q[9];
barrier q;

cx q[5], q[8];
cx q[2], q[7];
barrier q;

measure q[8] -> rec[15];
measure q[7] -> rec[16];
measure q[9] -> rec[17];
barrier q;

reset q[8];
reset q[7];
reset q[9];
h q[8];
h q[7];
h q[9];
barrier q;

cx q[7], q[0];
cx q[9], q[3];
barrier q;

cx q[7], q[1];
cx q[9], q[5];
barrier q;

cx q[8], q[1];
cx q[9], q[2];
barrier q;

cx q[8], q[6];
cx q[7], q[3];
barrier q;

cx q[8], q[3];
cx q[9], q[4];
barrier q;

cx q[8], q[5];
cx q[7], q[2];
barrier q;

h q[8];
h q[7];
h q[9];
measure q[8] -> rec[18];
measure q[7] -> rec[19];
measure q[9] -> rec[20];
h q[8];
h q[7];
h q[9];
barrier q;

reset q[8];
reset q[7];
reset q[9];
barrier q;

cx q[0], q[7];
cx q[3], q[9];
barrier q;

cx q[1], q[7];
cx q[5], q[9];
barrier q;

cx q[1], q[8];
cx q[2], q[9];
barrier q;

cx q[6], q[8];
cx q[3], q[7];
barrier q;

cx q[3], q[8];
cx q[4], q[9];
barrier q;

cx q[5], q[8];
cx q[2], q[7];
barrier q;

measure q[8] -> rec[21];
measure q[7] -> rec[22];
measure q[9] -> rec[23];
barrier q;

reset q[8];
reset q[7];
reset q[9];
h q[8];
h q[7];
h q[9];
barrier q;

cx q[7], q[0];
cx q[9], q[3];
barrier q;

cx q[7], q[1];
cx q[9], q[5];
barrier q;

cx q[8], q[1];
cx q[9], q[2];
barrier q;

cx q[8], q[6];
cx q[7], q[3];
barrier q;

cx q[8], q[3];
cx q[9], q[4];
barrier q;

cx q[8], q[5];
cx q[7], q[2];
barrier q;

h q[8];
h q[7];
h q[9];
measure q[8] -> rec[24];
measure q[7] -> rec[25];
measure q[9] -> rec[26];
h q[8];
h q[7];
h q[9];
barrier q;

reset q[8];
reset q[7];
reset q[9];
barrier q;

cx q[0], q[7];
cx q[3], q[9];
barrier q;

cx q[1], q[7];
cx q[5], q[9];
barrier q;

cx q[1], q[8];
cx q[2], q[9];
barrier q;

cx q[6], q[8];
cx q[3], q[7];
barrier q;

cx q[3], q[8];
cx q[4], q[9];
barrier q;

cx q[5], q[8];
cx q[2], q[7];
barrier q;

measure q[8] -> rec[27];
measure q[7] -> rec[28];
measure q[9] -> rec[29];
barrier q;

measure q[0] -> rec[30];
measure q[1] -> rec[31];
measure q[2] -> rec[32];
measure q[3] -> rec[33];
measure q[4] -> rec[34];
measure q[5] -> rec[35];
measure q[6] -> rec[36];