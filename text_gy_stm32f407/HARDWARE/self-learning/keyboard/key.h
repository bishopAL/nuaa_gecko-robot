#ifndef __KEY_H
#define __KEY_H

#include "sys.h"

//�˿ڶ�ȡ���define
#define WK_UP GPIO_ReadInputDataBit(GPIOA,GPIO_Pin_0) 
#define KEY_1 GPIO_ReadInputDataBit(GPIOE,GPIO_Pin_3) 
#define KEY_0 GPIO_ReadInputDataBit(GPIOE,GPIO_Pin_4) 
#define WK_UP_PRESS 2
#define KEY_0_PRESS 0
#define KEY_1_PRESS 1
#define ALL_KEY_UP 3
void KEY_Init(void);//��ʼ��
uint8_t KEY_Scan(void);
#endif